#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct HuffmanCode {
    uint16_t code;
    uint8_t length;
    uint8_t symbol;
};

struct HuffmanTableSpec {
    uint8_t table_class = 0;  // 0=DC, 1=AC
    uint8_t table_id = 0;
    std::vector<uint8_t> counts;   // size 16
    std::vector<uint8_t> symbols;  // sum(counts)
    std::vector<HuffmanCode> codes;
};

struct QuantTableSpec {
    uint8_t table_id = 0;
    uint8_t precision = 0;
    std::vector<uint16_t> values;
};

struct FrameComponentSpec {
    uint8_t component_id = 0;
    uint8_t h = 1;
    uint8_t v = 1;
    uint8_t tq = 0;
};

struct ScanComponentSpec {
    uint8_t component_id = 0;
    uint8_t dc_table = 0;
    uint8_t ac_table = 0;
    int frame_index = -1;
};

struct JpegSyntaxHeader {
    uint16_t width = 0;
    uint16_t height = 0;
    uint8_t precision = 8;
    uint16_t restart_interval = 0;
    std::vector<QuantTableSpec> quant_tables;
    std::vector<HuffmanTableSpec> huffman_tables;
    std::vector<FrameComponentSpec> frame_components;
    std::vector<ScanComponentSpec> scan_components;
    size_t entropy_start_offset = 0;
    size_t entropy_end_offset = 0;
};

struct BitEvent {
    int bit_offset = 0;
    int bit_length = 0;
    int value = 0;
};

struct TokenRecord {
    std::string token_type;
    int symbol = 0;
    int run = 0;
    int size = 0;
    int value = 0;
    int coeff_index = -1;
    BitEvent symbol_bits;
    std::optional<BitEvent> amplitude_bits;
};

struct BlockRecord {
    int block_in_mcu = 0;
    int component_index = 0;
    int component_id = 0;
    int dc_predictor_before = 0;
    int dc_predictor_after = 0;
    int block_bits_start = 0;
    int block_bits_end = 0;
    std::vector<TokenRecord> tokens;
};

struct McuRecord {
    int mcu_index = 0;
    int mcu_x = 0;
    int mcu_y = 0;
    int bit_start = 0;
    int bit_end = 0;
    std::vector<BlockRecord> blocks;
};

std::vector<uint8_t> read_file_bytes(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Unable to open JPEG file: " + path);
    }
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

std::vector<uint8_t> read_memory_bytes(const py::bytes &jpeg_bytes) {
    char *buffer = nullptr;
    Py_ssize_t size = 0;
    if (PyBytes_AsStringAndSize(jpeg_bytes.ptr(), &buffer, &size) != 0) {
        throw py::error_already_set();
    }
    auto *begin = reinterpret_cast<const uint8_t *>(buffer);
    return std::vector<uint8_t>(begin, begin + size);
}

uint16_t read_be16(const std::vector<uint8_t> &buf, size_t pos) {
    if (pos + 1 >= buf.size()) throw std::runtime_error("Unexpected EOF while reading uint16");
    return (static_cast<uint16_t>(buf[pos]) << 8) | static_cast<uint16_t>(buf[pos + 1]);
}

void build_huffman_codes(HuffmanTableSpec &table) {
    table.codes.clear();
    uint16_t code = 0;
    size_t symbol_pos = 0;
    for (int i = 0; i < 16; ++i) {
        int count = table.counts.at(i);
        for (int j = 0; j < count; ++j) {
            HuffmanCode hc;
            hc.code = code;
            hc.length = static_cast<uint8_t>(i + 1);
            hc.symbol = table.symbols.at(symbol_pos++);
            table.codes.push_back(hc);
            ++code;
        }
        code <<= 1;
    }
}

JpegSyntaxHeader parse_jpeg_header(const std::vector<uint8_t> &buf) {
    if (buf.size() < 4 || buf[0] != 0xFF || buf[1] != 0xD8) {
        throw std::runtime_error("Input is not a valid JPEG (missing SOI)");
    }

    JpegSyntaxHeader out;
    size_t pos = 2;

    while (pos < buf.size()) {
        while (pos < buf.size() && buf[pos] != 0xFF) ++pos;
        if (pos >= buf.size()) break;
        while (pos < buf.size() && buf[pos] == 0xFF) ++pos;
        if (pos >= buf.size()) break;

        uint8_t marker = buf[pos++];
        if (marker == 0xD9) {
            break;
        }
        if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
            continue;
        }

        if (pos + 1 >= buf.size()) throw std::runtime_error("Malformed JPEG marker length");
        uint16_t seglen = read_be16(buf, pos);
        pos += 2;
        if (seglen < 2 || pos + (seglen - 2) > buf.size()) {
            throw std::runtime_error("Malformed JPEG segment length");
        }
        size_t seg_start = pos;
        size_t seg_end = pos + (seglen - 2);

        if (marker == 0xDB) {
            size_t p = seg_start;
            while (p < seg_end) {
                uint8_t pq_tq = buf[p++];
                QuantTableSpec qt;
                qt.precision = (pq_tq >> 4) & 0x0F;
                qt.table_id = pq_tq & 0x0F;
                int n = qt.precision ? 128 : 64;
                qt.values.reserve(64);
                if (qt.precision == 0) {
                    for (int i = 0; i < 64; ++i) qt.values.push_back(buf[p++]);
                } else {
                    for (int i = 0; i < 64; ++i) {
                        qt.values.push_back(read_be16(buf, p));
                        p += 2;
                    }
                }
                if ((qt.precision == 0 && n != 64) || (qt.precision == 1 && n != 128)) {
                    throw std::runtime_error("Unexpected DQT precision");
                }
                out.quant_tables.push_back(std::move(qt));
            }
        } else if (marker == 0xC4) {
            size_t p = seg_start;
            while (p < seg_end) {
                HuffmanTableSpec ht;
                uint8_t tc_th = buf[p++];
                ht.table_class = (tc_th >> 4) & 0x0F;
                ht.table_id = tc_th & 0x0F;
                ht.counts.resize(16);
                int total = 0;
                for (int i = 0; i < 16; ++i) {
                    ht.counts[i] = buf[p++];
                    total += ht.counts[i];
                }
                ht.symbols.reserve(total);
                for (int i = 0; i < total; ++i) ht.symbols.push_back(buf[p++]);
                build_huffman_codes(ht);
                out.huffman_tables.push_back(std::move(ht));
            }
        } else if (marker == 0xC0) {
            out.precision = buf[seg_start];
            out.height = read_be16(buf, seg_start + 1);
            out.width = read_be16(buf, seg_start + 3);
            uint8_t ncomp = buf[seg_start + 5];
            out.frame_components.clear();
            size_t p = seg_start + 6;
            for (int i = 0; i < ncomp; ++i) {
                FrameComponentSpec fc;
                fc.component_id = buf[p++];
                uint8_t hv = buf[p++];
                fc.h = (hv >> 4) & 0x0F;
                fc.v = hv & 0x0F;
                fc.tq = buf[p++];
                out.frame_components.push_back(fc);
            }
        } else if (marker == 0xDD) {
            out.restart_interval = read_be16(buf, seg_start);
        } else if (marker == 0xDA) {
            uint8_t nscan = buf[seg_start];
            out.scan_components.clear();
            size_t p = seg_start + 1;
            for (int i = 0; i < nscan; ++i) {
                ScanComponentSpec sc;
                sc.component_id = buf[p++];
                uint8_t tdta = buf[p++];
                sc.dc_table = (tdta >> 4) & 0x0F;
                sc.ac_table = tdta & 0x0F;
                for (size_t k = 0; k < out.frame_components.size(); ++k) {
                    if (out.frame_components[k].component_id == sc.component_id) {
                        sc.frame_index = static_cast<int>(k);
                        break;
                    }
                }
                out.scan_components.push_back(sc);
            }
            out.entropy_start_offset = seg_end;

            size_t q = seg_end;
            while (q + 1 < buf.size()) {
                if (buf[q] == 0xFF && buf[q + 1] != 0x00) {
                    uint8_t next = buf[q + 1];
                    if (next == 0xD9 || (next >= 0xD0 && next <= 0xD7)) {
                        out.entropy_end_offset = q;
                        break;
                    }
                }
                ++q;
            }
            if (out.entropy_end_offset == 0) out.entropy_end_offset = buf.size();
            break;
        }

        pos = seg_end;
    }

    return out;
}

const HuffmanTableSpec* find_huffman_table(const JpegSyntaxHeader &hdr, int table_class, int table_id) {
    for (const auto &ht : hdr.huffman_tables) {
        if (ht.table_class == table_class && ht.table_id == table_id) return &ht;
    }
    return nullptr;
}

class EntropyBitReader {
public:
    EntropyBitReader(const std::vector<uint8_t> &buf, size_t start, size_t end)
        : buf_(buf), pos_(start), end_(end) {}

    int current_bit_offset() const { return bit_offset_; }

    bool read_bit(int &bit) {
        if (bits_left_ == 0) {
            if (!fill_byte()) return false;
        }
        bit = (current_byte_ >> (bits_left_ - 1)) & 1;
        --bits_left_;
        ++bit_offset_;
        return true;
    }

    int read_bits(int n) {
        int v = 0;
        for (int i = 0; i < n; ++i) {
            int b;
            if (!read_bit(b)) throw std::runtime_error("Unexpected end of entropy stream");
            v = (v << 1) | b;
        }
        return v;
    }

private:
    bool fill_byte() {
        while (pos_ < end_) {
            uint8_t b = buf_[pos_++];
            if (b == 0xFF) {
                if (pos_ >= end_) return false;
                uint8_t next = buf_[pos_];
                if (next == 0x00) {
                    ++pos_;
                    current_byte_ = 0xFF;
                    bits_left_ = 8;
                    return true;
                }
                if (next >= 0xD0 && next <= 0xD7) {
                    ++pos_;
                    bits_left_ = 0;
                    continue;
                }
                return false;
            }
            current_byte_ = b;
            bits_left_ = 8;
            return true;
        }
        return false;
    }

    const std::vector<uint8_t> &buf_;
    size_t pos_;
    size_t end_;
    uint8_t current_byte_ = 0;
    int bits_left_ = 0;
    int bit_offset_ = 0;
};

TokenRecord decode_huffman_symbol(EntropyBitReader &br, const HuffmanTableSpec &ht, const std::string &token_type) {
    int code = 0;
    int start = br.current_bit_offset();
    for (int len = 1; len <= 16; ++len) {
        int b;
        if (!br.read_bit(b)) throw std::runtime_error("Unexpected end while decoding Huffman symbol");
        code = (code << 1) | b;
        for (const auto &hc : ht.codes) {
            if (hc.length == len && hc.code == code) {
                TokenRecord tr;
                tr.token_type = token_type;
                tr.symbol = hc.symbol;
                tr.symbol_bits = BitEvent{start, len, code};
                return tr;
            }
        }
    }
    throw std::runtime_error("Failed to decode Huffman symbol");
}

int extend_signed_value(int bits, int size) {
    if (size == 0) return 0;
    int vt = 1 << (size - 1);
    if (bits < vt) return bits - ((1 << size) - 1);
    return bits;
}

py::dict bit_event_to_py(const BitEvent &e) {
    py::dict d;
    d["bit_offset"] = e.bit_offset;
    d["bit_length"] = e.bit_length;
    d["value"] = e.value;
    return d;
}

py::dict token_to_py(const TokenRecord &t) {
    py::dict d;
    d["token_type"] = t.token_type;
    d["symbol"] = t.symbol;
    d["run"] = t.run;
    d["size"] = t.size;
    d["value"] = t.value;
    d["coeff_index"] = t.coeff_index;
    d["symbol_bits"] = bit_event_to_py(t.symbol_bits);
    if (t.amplitude_bits.has_value()) {
        d["amplitude_bits"] = bit_event_to_py(*t.amplitude_bits);
    } else {
        d["amplitude_bits"] = py::none();
    }
    return d;
}

py::dict block_to_py(const BlockRecord &b) {
    py::list toks;
    for (const auto &t : b.tokens) toks.append(token_to_py(t));
    py::dict d;
    d["block_in_mcu"] = b.block_in_mcu;
    d["component_index"] = b.component_index;
    d["component_id"] = b.component_id;
    d["dc_predictor_before"] = b.dc_predictor_before;
    d["dc_predictor_after"] = b.dc_predictor_after;
    d["block_bits_start"] = b.block_bits_start;
    d["block_bits_end"] = b.block_bits_end;
    d["tokens"] = toks;
    return d;
}

py::dict mcu_to_py(const McuRecord &m) {
    py::list blocks;
    for (const auto &b : m.blocks) blocks.append(block_to_py(b));
    py::dict d;
    d["mcu_index"] = m.mcu_index;
    d["mcu_x"] = m.mcu_x;
    d["mcu_y"] = m.mcu_y;
    d["bit_start"] = m.bit_start;
    d["bit_end"] = m.bit_end;
    d["blocks"] = blocks;
    return d;
}

std::vector<uint8_t> extract_entropy_data_bytes(const std::vector<uint8_t> &buf, size_t start, size_t end) {
    std::vector<uint8_t> out;
    size_t pos = start;
    while (pos < end) {
        uint8_t b = buf[pos++];
        if (b == 0xFF) {
            if (pos >= end) break;
            uint8_t next = buf[pos];
            if (next == 0x00) {
                ++pos;
                out.push_back(0xFF);
                continue;
            }
            if (next >= 0xD0 && next <= 0xD7) {
                ++pos;
                continue;
            }
            break;
        }
        out.push_back(b);
    }
    return out;
}

std::vector<uint8_t> slice_bits_to_packed_bytes(const std::vector<uint8_t> &entropy_data, int bit_start, int bit_end) {
    if (bit_start < 0 || bit_end < bit_start) {
        throw std::runtime_error("Invalid bit range for MCU slice");
    }
    int valid_bits = bit_end - bit_start;
    std::vector<uint8_t> out(static_cast<size_t>((valid_bits + 7) / 8), 0);
    for (int i = 0; i < valid_bits; ++i) {
        int src_bit = bit_start + i;
        uint8_t src_byte = entropy_data.at(static_cast<size_t>(src_bit / 8));
        int src_bit_in_byte = 7 - (src_bit % 8);
        int bit = (src_byte >> src_bit_in_byte) & 1;
        size_t dst_byte = static_cast<size_t>(i / 8);
        int dst_bit_in_byte = 7 - (i % 8);
        out[dst_byte] |= static_cast<uint8_t>(bit << dst_bit_in_byte);
    }
    return out;
}

py::dict packed_mcu_to_py(const McuRecord &m, const std::vector<uint8_t> &payload_bytes) {
    py::dict d;
    int valid_bits = m.bit_end - m.bit_start;
    int byte_len = static_cast<int>(payload_bytes.size());
    int tail_bits = 0;
    if (valid_bits > 0) {
        tail_bits = valid_bits % 8;
        if (tail_bits == 0) tail_bits = 8;
    }
    d["mcu_index"] = m.mcu_index;
    d["mcu_x"] = m.mcu_x;
    d["mcu_y"] = m.mcu_y;
    d["bit_start"] = m.bit_start;
    d["bit_end"] = m.bit_end;
    d["valid_bits"] = valid_bits;
    d["byte_len"] = byte_len;
    d["tail_bits"] = tail_bits;
    d["payload_bytes"] = py::bytes(reinterpret_cast<const char*>(payload_bytes.data()), payload_bytes.size());
    py::list payload_list;
    for (uint8_t b : payload_bytes) payload_list.append(py::int_(b));
    d["payload_list"] = payload_list;
    return d;
}

py::dict header_to_py(const JpegSyntaxHeader &h) {
    py::dict d;
    d["width"] = h.width;
    d["height"] = h.height;
    d["precision"] = h.precision;
    d["restart_interval"] = h.restart_interval;
    d["entropy_start_offset"] = py::int_(h.entropy_start_offset);
    d["entropy_end_offset"] = py::int_(h.entropy_end_offset);

    py::list quant_tables;
    for (const auto &qt : h.quant_tables) {
        py::dict q;
        q["table_id"] = qt.table_id;
        q["precision"] = qt.precision;
        py::list values;
        for (auto v : qt.values) values.append(py::int_(v));
        q["values"] = values;
        quant_tables.append(q);
    }
    d["quant_tables"] = quant_tables;

    py::list frame_components;
    for (const auto &fc : h.frame_components) {
        py::dict c;
        c["component_id"] = fc.component_id;
        c["h"] = fc.h;
        c["v"] = fc.v;
        c["tq"] = fc.tq;
        frame_components.append(c);
    }
    d["frame_components"] = frame_components;

    py::list scan_components;
    for (const auto &sc : h.scan_components) {
        py::dict c;
        c["component_id"] = sc.component_id;
        c["dc_table"] = sc.dc_table;
        c["ac_table"] = sc.ac_table;
        c["frame_index"] = sc.frame_index;
        scan_components.append(c);
    }
    d["scan_components"] = scan_components;

    return d;
}

struct ParsedMcuSyntax {
    JpegSyntaxHeader header;
    int mcus_y;
    int mcus_x;
    std::vector<McuRecord> mcus;
};

ParsedMcuSyntax parse_mcu_syntax_from_bytes(const std::vector<uint8_t> &bytes) {
    ParsedMcuSyntax out;
    out.header = parse_jpeg_header(bytes);
    const auto &hdr = out.header;

    if (hdr.scan_components.empty() || hdr.frame_components.empty()) {
        throw std::runtime_error("JPEG scan/frame metadata incomplete; only baseline sequential JPEG is currently supported");
    }

    int max_h = 1;
    int max_v = 1;
    for (const auto &fc : hdr.frame_components) {
        max_h = std::max<int>(max_h, fc.h);
        max_v = std::max<int>(max_v, fc.v);
    }

    int mcu_width = 8 * max_h;
    int mcu_height = 8 * max_v;
    out.mcus_x = (hdr.width + mcu_width - 1) / mcu_width;
    out.mcus_y = (hdr.height + mcu_height - 1) / mcu_height;

    std::vector<int> dc_predictors(hdr.frame_components.size(), 0);
    EntropyBitReader br(bytes, hdr.entropy_start_offset, hdr.entropy_end_offset);
    out.mcus.reserve(static_cast<size_t>(out.mcus_x * out.mcus_y));
    int global_mcu_idx = 0;

    for (int my = 0; my < out.mcus_y; ++my) {
        for (int mx = 0; mx < out.mcus_x; ++mx) {
            McuRecord mcu;
            mcu.mcu_index = global_mcu_idx++;
            mcu.mcu_x = mx;
            mcu.mcu_y = my;
            mcu.bit_start = br.current_bit_offset();

            for (const auto &sc : hdr.scan_components) {
                const auto &fc = hdr.frame_components.at(sc.frame_index);
                const HuffmanTableSpec *dc_ht = find_huffman_table(hdr, 0, sc.dc_table);
                const HuffmanTableSpec *ac_ht = find_huffman_table(hdr, 1, sc.ac_table);
                if (!dc_ht || !ac_ht) throw std::runtime_error("Missing Huffman table for scan component");

                int blocks_in_component_mcu = fc.h * fc.v;
                for (int bi = 0; bi < blocks_in_component_mcu; ++bi) {
                    BlockRecord block;
                    block.block_in_mcu = bi;
                    block.component_index = sc.frame_index;
                    block.component_id = sc.component_id;
                    block.dc_predictor_before = dc_predictors.at(sc.frame_index);
                    block.block_bits_start = br.current_bit_offset();

                    auto dc_tok = decode_huffman_symbol(br, *dc_ht, "DC");
                    dc_tok.size = dc_tok.symbol;
                    dc_tok.run = 0;
                    dc_tok.coeff_index = 0;
                    if (dc_tok.size > 0) {
                        int amp_start = br.current_bit_offset();
                        int raw = br.read_bits(dc_tok.size);
                        dc_tok.amplitude_bits = BitEvent{amp_start, dc_tok.size, raw};
                        dc_tok.value = extend_signed_value(raw, dc_tok.size);
                    }
                    dc_predictors.at(sc.frame_index) += dc_tok.value;
                    block.dc_predictor_after = dc_predictors.at(sc.frame_index);
                    block.tokens.push_back(dc_tok);

                    int k = 1;
                    while (k < 64) {
                        auto ac_tok = decode_huffman_symbol(br, *ac_ht, "AC");
                        int symbol = ac_tok.symbol;
                        ac_tok.run = (symbol >> 4) & 0x0F;
                        ac_tok.size = symbol & 0x0F;

                        if (symbol == 0x00) {
                            ac_tok.token_type = "EOB";
                            ac_tok.coeff_index = k;
                            block.tokens.push_back(ac_tok);
                            break;
                        }
                        if (symbol == 0xF0) {
                            ac_tok.token_type = "ZRL";
                            ac_tok.coeff_index = k;
                            k += 16;
                            block.tokens.push_back(ac_tok);
                            continue;
                        }

                        k += ac_tok.run;
                        ac_tok.coeff_index = k;
                        if (ac_tok.size > 0) {
                            int amp_start = br.current_bit_offset();
                            int raw = br.read_bits(ac_tok.size);
                            ac_tok.amplitude_bits = BitEvent{amp_start, ac_tok.size, raw};
                            ac_tok.value = extend_signed_value(raw, ac_tok.size);
                        }
                        ++k;
                        block.tokens.push_back(ac_tok);
                    }

                    block.block_bits_end = br.current_bit_offset();
                    mcu.blocks.push_back(std::move(block));
                }
            }

            mcu.bit_end = br.current_bit_offset();
            out.mcus.push_back(std::move(mcu));
        }
    }

    return out;
}

py::dict syntax_to_py(const ParsedMcuSyntax &syntax) {
    py::list mcus_py;
    for (const auto &mcu : syntax.mcus) mcus_py.append(mcu_to_py(mcu));

    py::dict out;
    out["header"] = header_to_py(syntax.header);
    out["mcu_grid"] = py::make_tuple(syntax.mcus_y, syntax.mcus_x);
    out["mcu_count"] = syntax.mcus_x * syntax.mcus_y;
    out["mcus"] = mcus_py;
    return out;
}

py::dict read_syntax_patches_from_bytes(const std::vector<uint8_t> &bytes) {
    return syntax_to_py(parse_mcu_syntax_from_bytes(bytes));
}

py::dict read_syntax_patches(const std::string &path) {
    return read_syntax_patches_from_bytes(read_file_bytes(path));
}

py::dict read_syntax_patches_from_buffer(const py::bytes &jpeg_bytes) {
    return read_syntax_patches_from_bytes(read_memory_bytes(jpeg_bytes));
}

py::dict read_mcu_packed_bytes_from_bytes(const std::vector<uint8_t> &bytes) {
    auto syntax = parse_mcu_syntax_from_bytes(bytes);
    auto entropy_data = extract_entropy_data_bytes(bytes, syntax.header.entropy_start_offset, syntax.header.entropy_end_offset);

    py::list packed_mcus;
    for (const auto &rec : syntax.mcus) {
        auto payload_bytes = slice_bits_to_packed_bytes(entropy_data, rec.bit_start, rec.bit_end);
        packed_mcus.append(packed_mcu_to_py(rec, payload_bytes));
    }

    py::dict out;
    out["header"] = header_to_py(syntax.header);
    out["mcu_grid"] = py::make_tuple(syntax.mcus_y, syntax.mcus_x);
    out["mcu_count"] = syntax.mcus_x * syntax.mcus_y;
    out["mcus"] = packed_mcus;
    return out;
}

torch::Tensor read_mcu_payload_tensor_from_bytes(const std::vector<uint8_t> &bytes, int lmax) {
    if (lmax <= 0) {
        throw std::runtime_error("lmax must be positive");
    }

    auto syntax = parse_mcu_syntax_from_bytes(bytes);
    auto entropy_data = extract_entropy_data_bytes(bytes, syntax.header.entropy_start_offset, syntax.header.entropy_end_offset);
    auto payload_tensor = torch::zeros({static_cast<long>(syntax.mcus.size()), lmax}, torch::dtype(torch::kUInt8));
    auto payload_acc = payload_tensor.accessor<uint8_t, 2>();

    for (size_t i = 0; i < syntax.mcus.size(); ++i) {
        const auto &rec = syntax.mcus[i];
        auto payload_bytes = slice_bits_to_packed_bytes(entropy_data, rec.bit_start, rec.bit_end);
        size_t n = std::min(payload_bytes.size(), static_cast<size_t>(lmax));
        for (size_t j = 0; j < n; ++j) {
            payload_acc[static_cast<long>(i)][static_cast<long>(j)] = payload_bytes[j];
        }
    }

    return payload_tensor;
}

py::dict read_mcu_packed_bytes(const std::string &path) {
    return read_mcu_packed_bytes_from_bytes(read_file_bytes(path));
}

torch::Tensor read_mcu_payload_tensor(const std::string &path, int lmax) {
    return read_mcu_payload_tensor_from_bytes(read_file_bytes(path), lmax);
}

py::dict read_mcu_packed_bytes_from_buffer(const py::bytes &jpeg_bytes) {
    return read_mcu_packed_bytes_from_bytes(read_memory_bytes(jpeg_bytes));
}

torch::Tensor read_mcu_payload_tensor_from_buffer(const py::bytes &jpeg_bytes, int lmax) {
    return read_mcu_payload_tensor_from_bytes(read_memory_bytes(jpeg_bytes), lmax);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace pybind11::literals;
    m.def("read_syntax_patches", &read_syntax_patches,
          "Read MCU-level JPEG syntax patches directly from the JPEG entropy stream",
          "path"_a);
    m.def("read_syntax_patches_from_buffer", &read_syntax_patches_from_buffer,
          "Read MCU-level JPEG syntax patches directly from an in-memory JPEG byte buffer",
          "jpeg_bytes"_a);
    m.def("read_mcu_packed_bytes", &read_mcu_packed_bytes,
          "Read MCU-level packed-byte payloads from exact JPEG entropy bit ranges",
          "path"_a);
    m.def("read_mcu_payload_tensor", &read_mcu_payload_tensor,
          "Read MCU-level packed-byte payloads directly into a uint8 tensor of shape [mcu_count, lmax]",
          "path"_a, "lmax"_a);
    m.def("read_mcu_packed_bytes_from_buffer", &read_mcu_packed_bytes_from_buffer,
          "Read MCU-level packed-byte payloads from exact JPEG entropy bit ranges in an in-memory JPEG byte buffer",
          "jpeg_bytes"_a);
    m.def("read_mcu_payload_tensor_from_buffer", &read_mcu_payload_tensor_from_buffer,
          "Read MCU-level packed-byte payloads directly into a uint8 tensor of shape [mcu_count, lmax] from an in-memory JPEG byte buffer",
          "jpeg_bytes"_a, "lmax"_a);
}
