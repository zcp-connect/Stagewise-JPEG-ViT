#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace jpeg_payload_extract {

struct HuffmanCode {
    uint16_t code;
    uint8_t length;
    uint8_t symbol;
};

struct HuffmanTableSpec {
    uint8_t table_class = 0;
    uint8_t table_id = 0;
    std::vector<uint8_t> counts;
    std::vector<uint8_t> symbols;
    std::vector<HuffmanCode> codes;
    
    // Fast lookup table optimization
    static constexpr int MAX_PREFIX_LEN = 12;  // Use 12-bit prefix for LUT
    static constexpr int LUT_SIZE = 1 << MAX_PREFIX_LEN;  // 4096 entries
    struct LutEntry {
        uint8_t symbol = 0xFF;      // 0xFF = invalid/not in table
        uint8_t code_len = 0;       // actual code length (1-16)
    };
    std::vector<LutEntry> lut;      // Fast lookup table
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
    std::vector<HuffmanTableSpec> huffman_tables;
    std::vector<FrameComponentSpec> frame_components;
    std::vector<ScanComponentSpec> scan_components;
    size_t entropy_start_offset = 0;
    size_t entropy_end_offset = 0;
};

struct McuBitRange {
    int mcu_index = 0;
    int mcu_x = 0;
    int mcu_y = 0;
    int bit_start = 0;
    int bit_end = 0;
};

struct ParsedMcuPayloadSyntax {
    JpegSyntaxHeader header;
    int mcus_y = 0;
    int mcus_x = 0;
    std::vector<McuBitRange> mcus;
};

inline uint16_t read_be16(const std::vector<uint8_t> &buf, size_t pos) {
    if (pos + 1 >= buf.size()) throw std::runtime_error("Unexpected EOF while reading uint16");
    return (static_cast<uint16_t>(buf[pos]) << 8) | static_cast<uint16_t>(buf[pos + 1]);
}

inline void build_huffman_codes(HuffmanTableSpec &table) {
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
    
    // Build fast lookup table for codes up to MAX_PREFIX_LEN bits
    table.lut.assign(HuffmanTableSpec::LUT_SIZE, {0xFF, 0});
    for (const auto &hc : table.codes) {
        if (hc.length <= HuffmanTableSpec::MAX_PREFIX_LEN) {
            // Left-align the code to MAX_PREFIX_LEN bits
            uint16_t aligned_code = static_cast<uint16_t>(hc.code) << (HuffmanTableSpec::MAX_PREFIX_LEN - hc.length);
            uint16_t mask = (1U << (HuffmanTableSpec::MAX_PREFIX_LEN - hc.length)) - 1;
            
            // Fill all entries that match this prefix
            for (uint16_t i = 0; i <= mask; ++i) {
                uint16_t idx = aligned_code | i;
                if (idx < HuffmanTableSpec::LUT_SIZE) {
                    table.lut[idx] = {hc.symbol, hc.length};
                }
            }
        }
    }
}

inline JpegSyntaxHeader parse_jpeg_header(const std::vector<uint8_t> &buf) {
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

        if (marker == 0xC4) {
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

inline const HuffmanTableSpec* find_huffman_table(const JpegSyntaxHeader &hdr, int table_class, int table_id) {
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

inline uint8_t decode_huffman_symbol(EntropyBitReader &br, const HuffmanTableSpec &ht) {
    int code = 0;
    for (int len = 1; len <= 16; ++len) {
        int b;
        if (!br.read_bit(b)) throw std::runtime_error("Unexpected end while decoding Huffman symbol");
        code = (code << 1) | b;
        for (const auto &hc : ht.codes) {
            if (hc.length == len && hc.code == code) {
                return hc.symbol;
            }
        }
    }
    throw std::runtime_error("Failed to decode Huffman symbol");
}

inline int extend_signed_value(int bits, int size) {
    if (size == 0) return 0;
    int vt = 1 << (size - 1);
    if (bits < vt) return bits - ((1 << size) - 1);
    return bits;
}

inline std::vector<uint8_t> extract_entropy_data_bytes(const std::vector<uint8_t> &buf, size_t start, size_t end) {
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

inline std::vector<uint8_t> slice_bits_to_packed_bytes(const std::vector<uint8_t> &entropy_data, int bit_start, int bit_end) {
    if (bit_start < 0 || bit_end < bit_start) {
        throw std::runtime_error("Invalid bit range for MCU slice");
    }
    int valid_bits = bit_end - bit_start;
    std::vector<uint8_t> out(static_cast<size_t>((valid_bits + 7) / 8), 0);
    
    // Fast path: byte-aligned case (common for MCU data)
    // When bit_start is byte-aligned and valid_bits is multiple of 8
    if ((bit_start % 8) == 0 && (valid_bits % 8) == 0) {
        int src_byte_idx = bit_start / 8;
        int num_bytes = valid_bits / 8;
        if (src_byte_idx + num_bytes <= static_cast<int>(entropy_data.size())) {
            std::memcpy(out.data(), entropy_data.data() + src_byte_idx, num_bytes);
            return out;
        }
    }
    
    // Mid-path: byte-aligned extraction (start & length, but source bits may not align)
    // This is still more efficient than per-bit operations
    int start_byte = bit_start / 8;
    int start_bit_in_byte = bit_start % 8;
    
    if (start_bit_in_byte == 0) {
        // Source is byte-aligned but output might not require full bytes
        // Still more efficient: copy bytes then mask last byte if needed
        int full_bytes = valid_bits / 8;
        if (full_bytes > 0) {
            std::memcpy(out.data(), entropy_data.data() + start_byte, full_bytes);
        }
        
        // Handle remaining bits using byte extraction
        int remaining_bits = valid_bits % 8;
        if (remaining_bits > 0) {
            uint8_t last_byte = entropy_data.at(start_byte + full_bytes);
            uint8_t mask = static_cast<uint8_t>(0xFF << (8 - remaining_bits));
            out[full_bytes] = last_byte & mask;
        }
        return out;
    }
    
    // Slow path: general bit slicing (per-bit operation) - fallback for unaligned case
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


inline bool try_find_entropy_start_offset(const std::vector<uint8_t> &buf, size_t &entropy_start_offset) {
    if (buf.size() < 2) return false;
    if (buf[0] != 0xFF || buf[1] != 0xD8) {
        throw std::runtime_error("Input is not a valid JPEG prefix (missing SOI)");
    }

    size_t pos = 2;
    while (pos < buf.size()) {
        while (pos < buf.size() && buf[pos] != 0xFF) ++pos;
        if (pos >= buf.size()) return false;
        while (pos < buf.size() && buf[pos] == 0xFF) ++pos;
        if (pos >= buf.size()) return false;

        uint8_t marker = buf[pos++];
        if (marker == 0xD9) return false;
        if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
            continue;
        }

        if (pos + 1 >= buf.size()) return false;
        uint16_t seglen = read_be16(buf, pos);
        if (seglen < 2) {
            throw std::runtime_error("Malformed JPEG segment length");
        }
        if (pos + seglen > buf.size()) return false;
        pos += 2;
        size_t seg_end = pos + (seglen - 2);
        if (marker == 0xDA) {
            entropy_start_offset = seg_end;
            return true;
        }
        pos = seg_end;
    }
    return false;
}

inline void compute_mcu_grid(const JpegSyntaxHeader &hdr, int &mcus_x, int &mcus_y) {
    int max_h = 1;
    int max_v = 1;
    for (const auto &fc : hdr.frame_components) {
        max_h = std::max<int>(max_h, fc.h);
        max_v = std::max<int>(max_v, fc.v);
    }

    int mcu_width = 8 * max_h;
    int mcu_height = 8 * max_v;
    mcus_x = (hdr.width + mcu_width - 1) / mcu_width;
    mcus_y = (hdr.height + mcu_height - 1) / mcu_height;
}

class IncrementalEntropyBitReader {
public:
    explicit IncrementalEntropyBitReader(const std::vector<uint8_t> &buf)
        : buf_(buf) {}

    int current_bit_offset() const { return bit_offset_; }

    void copy_state_from(const IncrementalEntropyBitReader &other) {
        pos_ = other.pos_;
        current_byte_ = other.current_byte_;
        bits_left_ = other.bits_left_;
        bit_offset_ = other.bit_offset_;
    }

    bool read_bit(int &bit) {
        if (bits_left_ == 0) {
            if (pos_ >= buf_.size()) return false;
            current_byte_ = buf_[pos_++];
            bits_left_ = 8;
        }
        bit = (current_byte_ >> (bits_left_ - 1)) & 1;
        --bits_left_;
        ++bit_offset_;
        return true;
    }

    bool read_bits(int n, int &value) {
        value = 0;
        for (int i = 0; i < n; ++i) {
            int b;
            if (!read_bit(b)) return false;
            value = (value << 1) | b;
        }
        return true;
    }
    
    // Peek n bits without consuming them
    bool peek_bits(int n, int &value) const {
        if (n > 16 || n <= 0) return false;  // Limit to reasonable size
        IncrementalEntropyBitReader temp_reader(*this);
        return temp_reader.read_bits(n, value);
    }
    
    // Skip n bits
    bool skip_bits(int n) {
        for (int i = 0; i < n; ++i) {
            int dummy;
            if (!read_bit(dummy)) return false;
        }
        return true;
    }

private:
    const std::vector<uint8_t> &buf_;
    size_t pos_ = 0;
    uint8_t current_byte_ = 0;
    int bits_left_ = 0;
    int bit_offset_ = 0;
    
    // Allow copy_state_from to access private members
    friend class IncrementalEntropyBitReader;
};

// Fast Huffman decoding using lookup table
inline bool try_decode_huffman_symbol_fast(IncrementalEntropyBitReader &br, const HuffmanTableSpec &ht, uint8_t &symbol) {
    // Try LUT first for codes up to 12 bits
    int prefix_bits = 0;
    if (br.peek_bits(HuffmanTableSpec::MAX_PREFIX_LEN, prefix_bits)) {
        const auto &entry = ht.lut[prefix_bits];
        if (entry.symbol != 0xFF) {  // Found in LUT
            if (!br.skip_bits(entry.code_len)) return false;
            symbol = entry.symbol;
            return true;
        }
    }
    
    // Fallback to linear search for longer codes (rare)
    int code = 0;
    for (int len = 1; len <= 16; ++len) {
        int b;
        if (!br.read_bit(b)) return false;
        code = (code << 1) | b;
        for (const auto &hc : ht.codes) {
            if (hc.length == len && hc.code == code) {
                symbol = hc.symbol;
                return true;
            }
        }
    }
    throw std::runtime_error("Failed to decode Huffman symbol (fast path)");
}

inline bool try_decode_huffman_symbol(IncrementalEntropyBitReader &br, const HuffmanTableSpec &ht, uint8_t &symbol) {
    // Use fast path by default
    return try_decode_huffman_symbol_fast(br, ht, symbol);
}

inline bool try_decode_next_mcu(const JpegSyntaxHeader &hdr,
                                IncrementalEntropyBitReader &br,
                                int mcu_index,
                                int mcu_x,
                                int mcu_y,
                                McuBitRange &mcu) {
    if (hdr.scan_components.empty() || hdr.frame_components.empty()) {
        throw std::runtime_error("JPEG scan/frame metadata incomplete; only baseline sequential JPEG is currently supported");
    }

    mcu.mcu_index = mcu_index;
    mcu.mcu_x = mcu_x;
    mcu.mcu_y = mcu_y;
    mcu.bit_start = br.current_bit_offset();

    for (const auto &sc : hdr.scan_components) {
        const auto &fc = hdr.frame_components.at(sc.frame_index);
        const HuffmanTableSpec *dc_ht = find_huffman_table(hdr, 0, sc.dc_table);
        const HuffmanTableSpec *ac_ht = find_huffman_table(hdr, 1, sc.ac_table);
        if (!dc_ht || !ac_ht) throw std::runtime_error("Missing Huffman table for scan component");

        int blocks_in_component_mcu = fc.h * fc.v;
        for (int bi = 0; bi < blocks_in_component_mcu; ++bi) {
            uint8_t dc_symbol = 0;
            if (!try_decode_huffman_symbol(br, *dc_ht, dc_symbol)) return false;
            if (dc_symbol > 0) {
                int raw = 0;
                if (!br.read_bits(dc_symbol, raw)) return false;
            }

            int k = 1;
            while (k < 64) {
                uint8_t symbol = 0;
                if (!try_decode_huffman_symbol(br, *ac_ht, symbol)) return false;
                int run = (symbol >> 4) & 0x0F;
                int size = symbol & 0x0F;

                if (symbol == 0x00) {
                    break;
                }
                if (symbol == 0xF0) {
                    k += 16;
                    continue;
                }

                k += run;
                if (size > 0) {
                    int raw = 0;
                    if (!br.read_bits(size, raw)) return false;
                }
                ++k;
            }
        }
    }

    mcu.bit_end = br.current_bit_offset();
    return true;
}

inline ParsedMcuPayloadSyntax parse_mcu_payload_syntax_from_bytes(const std::vector<uint8_t> &bytes) {
    ParsedMcuPayloadSyntax out;
    out.header = parse_jpeg_header(bytes);
    const auto &hdr = out.header;

    if (hdr.scan_components.empty() || hdr.frame_components.empty()) {
        throw std::runtime_error("JPEG scan/frame metadata incomplete; only baseline sequential JPEG is currently supported");
    }

    compute_mcu_grid(hdr, out.mcus_x, out.mcus_y);

    std::vector<int> dc_predictors(hdr.frame_components.size(), 0);
    EntropyBitReader br(bytes, hdr.entropy_start_offset, hdr.entropy_end_offset);
    out.mcus.reserve(static_cast<size_t>(out.mcus_x * out.mcus_y));
    int global_mcu_idx = 0;

    for (int my = 0; my < out.mcus_y; ++my) {
        for (int mx = 0; mx < out.mcus_x; ++mx) {
            McuBitRange mcu;
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
                    uint8_t dc_symbol = decode_huffman_symbol(br, *dc_ht);
                    if (dc_symbol > 0) {
                        int raw = br.read_bits(dc_symbol);
                        dc_predictors.at(sc.frame_index) += extend_signed_value(raw, dc_symbol);
                    }

                    int k = 1;
                    while (k < 64) {
                        uint8_t symbol = decode_huffman_symbol(br, *ac_ht);
                        int run = (symbol >> 4) & 0x0F;
                        int size = symbol & 0x0F;

                        if (symbol == 0x00) {
                            break;
                        }
                        if (symbol == 0xF0) {
                            k += 16;
                            continue;
                        }

                        k += run;
                        if (size > 0) {
                            (void)br.read_bits(size);
                        }
                        ++k;
                    }
                }
            }

            mcu.bit_end = br.current_bit_offset();
            out.mcus.push_back(std::move(mcu));
        }
    }

    return out;
}

} // namespace jpeg_payload_extract
