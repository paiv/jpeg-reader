#!/usr/bin/env python
# author: Pavel paiv Ivashkov, 2019, https://github.com/paiv/jpeg-reader
import binascii
import io
import itertools
import math
import os
import pickle
import struct
import sys
from decimal import Decimal as dec

try:
    import numpy as np
except ImportError:
    np = None


def trace(*args, **kwargs):
    print(*args, flush=True, file=sys.stderr, **kwargs)


class JpegReaderError(Exception): pass


class JfifReader:
    class Error(Exception):
        pass

    def __init__(self, fp):
        self.iterator = JfifReader._Iterator(fp)

    def __iter__(self):
        return self.iterator

    class _Iterator:
        def __init__(self, fp):
            self.reader = io.BufferedReader(fp)
            self.parser = JfifReader._Parser()
            self.state = 0
            self.marker = None
            self.data = None
            self.data_to_read = None

        def __next__(self):
            return self._read_segment()

        def _read_segment(self):
            while True:
                obj = self._advance_reader()
                if obj: return obj

        def _peek1(self):
            xs = self.reader.peek(1)
            if not xs:
                pos = self.reader.tell()
                raise JfifReader.Error(f'Unexpected EOF at {pos}')
            return xs[0]

        def _read1(self):
            xs = self.reader.read1(1)
            if not xs:
                pos = self.reader.tell()
                raise JfifReader.Error(f'Unexpected EOF at {pos}')
            return xs[0]

        def _advance_reader(self):
            if self.state == 0:
                xs = self.reader.peek(1)
                if not xs:
                    raise StopIteration()
                x = xs[0]

                if x == 0xFF:
                    self.reader.read(1)
                    self.state = 1
                else:
                    pos = self.reader.tell()
                    raise JfifReader.Error(f'Corrupt data at {pos}: {x:02X}')

            elif self.state == 'DataScan':
                x = self._read1()
                if x == 0xFF:
                    self.state = 'DataScan-FF'
                else:
                    self.data.append(x)

            elif self.state == 'DataScan-FF':
                x = self._peek1()
                if x == 0:
                    self.data.append(0xFF)
                    self.reader.read(1)
                    self.state = 'DataScan'
                elif (x & 0xf8) == 0xD0: # RST
                    self.data.append(0xFF)
                    self.data.append(0xD0)
                    self.reader.read(1)
                    self.state = 'DataScan'
                else:
                    self.state = 1
                    return self._complete_segment()

            else:
                x = self._read1()

                if self.state == 1:
                    self.marker = [0xFF, x]

                    if x == 0xFF:
                        pass

                    elif x in (0xD8, 0xD9):
                        self.state = 0
                        return self._complete_segment()

                    elif x == 0xDA:
                        self.data = bytearray()
                        self.state = 'DataScan'

                    else:
                        self.state = 2

                elif self.state == 2:
                    self.data_to_read = x
                    self.state = 3

                elif self.state == 3:
                    self.data_to_read = self.data_to_read * 256 + x - 2
                    if self.data_to_read > 0:
                        self.data = bytearray()
                        self.state = 4
                    else:
                        self.state = 0
                        return self._complete_segment()

                elif self.state == 4:
                    self.data.append(x)
                    self.data_to_read -= 1
                    if self.data_to_read == 0:
                        self.state = 0
                        return self._complete_segment()

        def _complete_segment(self):
            (ma, mb), data = self.marker, self.data
            marker = (ma << 8) | mb
            data = bytes(data) if data else None
            self.marker = self.data = None
            self.data_to_read = None
            return self.parser.parse(marker, data)

    class _Parser:
        def __init__(self):
            self.seen_app0 = False

        def parse(self, marker, data):

            if marker == 0xFFD8:
                return JpegStartOfImage()

            elif marker == 0xFFD9:
                return JpegEndOfImage()

            elif marker == 0xFFE0:
                if self.seen_app0:
                    return JfifApp0XX.frombytes(data)
                self.seen_app0 = True
                return JfifApp0.frombytes(data)

            elif marker == 0xFFFE:
                return JpegComment.frombytes(data)

            else:
                return JpegSegment.frombytes(marker, data)


class JpegSegment:
    def __init__(self, marker):
        self.marker = marker
        self.data = None

    def __repr__(self):
        return f'{type(self).__name__}[{self.marker:04X}]'

    def frombytes(marker, data):
        obj = JpegSegment(marker)
        obj.data = data
        return obj


class JpegStartOfImage(JpegSegment):
    def __init__(self):
        super().__init__(0xFFD8)


class JpegEndOfImage(JpegSegment):
    def __init__(self):
        super().__init__(0xFFD9)


class JfifApp0(JpegSegment):
    def __init__(self):
        super().__init__(0xFFE0)

    def __repr__(self):
        s = super().__repr__()
        return f'{s}{{id:{self.identifier}, v:{self.version}, density:{self.density}, thumb:{self.thumbnail[:2]}}}'

    def frombytes(data):
        id, ver_maj, ver_min, units, densX, densY, thumbX, thumbY = struct.unpack('>5s3B2H2B', data[:14])
        rgb = data[14:]
        obj = JfifApp0()
        obj.identifier = id.decode('ascii')
        obj.version = '.'.join(map(str, (ver_maj, ver_min)))
        obj.density = (densX, densY, ['px', 'dpi', 'dpcm'][units] if units < 3 else None)
        obj.thumbnail = (thumbX, thumbY, rgb)
        return obj


class JfifApp0XX(JpegSegment):
    def __init__(self):
        super().__init__(0xFFE0)

    def frombytes(data):
        return JfifApp0XX()


class JpegComment(JpegSegment):
    def __init__(self):
        super().__init__(0xFFFE)

    def __repr__(self):
        s = super().__repr__()
        x = repr(self.text) if self.text is not None else 'hex:' + binascii.hexlify(self.data)
        return f'{s}{{{x}}}'

    def frombytes(data):
        obj = JpegComment()
        obj.data = data
        try:
            obj.text = data.decode('utf-8')
        except UnicodeError:
            pass
        return obj


class JpegRestartInterval(JpegSegment):
    def __init__(self):
        super().__init__(0xFFDD)

    def __repr__(self):
        s = super().__repr__()
        return f'{s}{{{self.size}}}'

    def frombytes(data):
        obj = JpegRestartInterval()
        obj.size, = struct.unpack('>H', data)
        return obj


class JpegQuantizationTables(JpegSegment):
    def __init__(self):
        super().__init__(0xFFDB)
        self.tables = list()

    def frombytes(data):
        obj = JpegQuantizationTables()
        off = 0
        while off < len(data):
            pt = data[off]
            precision = pt >> 4
            destination = pt & 0xf
            off += 1
            n = 64 * (precision + 1)
            qs = data[off:off+n]
            obj.tables.append(JpegImage.QuantizationTable(qs, destination=destination, precision=precision))
            off += n
        return obj


class JpegHuffmanTables(JpegSegment):
    def __init__(self):
        super().__init__(0xFFC4)
        self.tables = list()

    def frombytes(data):
        obj = JpegHuffmanTables()
        off = 0
        while off < len(data):
            pt = data[off]
            kind = pt >> 4
            destination = pt & 0xf
            off += 1
            ls = data[off:off+16]
            off += 16
            qs = [data[off+sum(ls[:i]):off+sum(ls[:i])+l] for i, l in enumerate(ls)]
            off += sum(ls)
            obj.tables.append(JpegImage.HuffmanTable(qs, destination=destination, kind=kind))
        return obj


class JpegStartOfFrame(JpegSegment):
    def __init__(self, marker):
        super().__init__(marker)

    def __repr__(self):
        s = super().__repr__()
        spec = ', '.join('{{{}, sampl:{} qt:{}}}'.format(c, (h, v), t) for c, h, v, t in self.spec)
        return f'{s}{{prec:{self.precision}, y:{self.y}, x:{self.x}, n:{self.n}, components:{{{spec}}}}}'

    def frombytes(data, marker=0xFFC0):
        obj = JpegStartOfFrame(marker)
        obj.precision, obj.y, obj.x, obj.n = struct.unpack('>B2HB', data[:6])
        i = 6
        spec = list()
        while i < len(data):
            c, hv, t = data[i:i+3]
            h = hv >> 4
            v = hv & 0xf
            spec.append((c, h, v, t))
            i += 3
        obj.spec = spec
        obj.components_iter = [c for (c, h, v, t) in spec for _ in range(h * v)]
        return obj


class JpegStartOfScan(JpegSegment):
    def __init__(self):
        super().__init__(0xFFDA)

    def __repr__(self):
        s = super().__repr__()
        spec = ', '.join(f'{{{c}, dct:{td}, act:{ta}}}' for c, td, ta in self.components)
        return f'{s}{{n:{self.n}, spec:{{{spec}}}, pro:({self.ss}, {self.se}, {self.ah}, {self.al}), len:{len(self.data)}}}'

    def frombytes(data):
        obj = JpegStartOfScan()
        size = struct.unpack('>H', data[:2])
        off = 2
        obj.n = data[off]
        off += 1
        spec = list()
        for _ in range(obj.n):
            sel, tda = struct.unpack('>BB', data[off:off+2])
            off += 2
            td = tda >> 4
            ta = tda & 0xf
            spec.append((sel, td, ta))
        obj.components = spec
        obj.ss, obj.se, ax = data[off:off+3]
        obj.ah = ax >> 4
        obj.al = ax & 0xf
        off += 3
        obj.data = data[off:]
        return obj


class BitStream:
    def __init__(self, data):
        self.data = data
        self.offset = 0
        self.bit = 7

    def __iter__(self):
        return self

    def __next__(self):
        if self.offset < len(self.data):
            v = self.data[self.offset]
            x = (v >> self.bit) & 1
            self.bit -= 1
            if self.bit < 0:
                self.bit = 7
                self.offset += 1
            return x
        else:
            raise StopIteration()

    def unpack(self, nbits):
        return sum((next(self) << (nbits - i - 1)) for i in range(nbits))


class HuffmanCodec:
    def __init__(self, table):
        root = HuffmanCodec._Node()
        fringe = [root.zero(), root.one()]
        for xs in table:
            nodes = iter(fringe)
            for x in xs:
                next(nodes).value = x
            fringe = [c for p in nodes for c in [p.zero(), p.one()]]
        self.table = root

    def __repr__(self):
        return f'HuffmanCodec{{{self.table}}}'

    def decode(self, data):
        t = type(data)
        return t(self._decode_stream(BitStream(data)))

    def feed(self, stream):
        p = self.table
        for x in stream:
            p = p.one() if x else p.zero()
            if p.value is not None:
                return p.value

    def _decode_stream(self, stream):
        while True:
            x = self.feed(stream)
            if x is None: break
            yield x

    class _Node:
        def __init__(self):
            self.z = None
            self.r = None
            self.value = None

        def __bool__(self):
            return not (self.value is None and not self.z and not self.r)

        def __repr__(self):
            if self.value is not None:
                return str(self.value)
            z = f'0:{{{repr(self.z)}}}' if self.z else ''
            r = f',1:{{{repr(self.r)}}}' if self.r else ''
            return z + r

        def zero(self):
            if self.z is None:
                self.z = HuffmanCodec._Node()
            return self.z

        def one(self):
            if self.r is None:
                self.r = HuffmanCodec._Node()
            return self.r


class JpegImage:
    def __init__(self):
        self.size = None
        self.jfif = None
        self.restart_interval = None
        self.quantization_tables = None # DQT
        self.huffman_tables = None # DHT
        self.unprocessed = list()

    class QuantizationTable:
        def __init__(self, data, destination=0, precision=1):
            self.data = data
            self.destination = destination
            self.precision = precision

        def __repr__(self):
            return f'QuantizationTable{{{self.destination}}}'

    class HuffmanTable:
        def __init__(self, data, destination=0, kind=0):
            self.data = data
            self.destination = destination
            self.kind = kind

        def __repr__(self):
            return f'HuffmanTable{{{self.destination}, kind:{self.kind}}}'


def _clamp(value, v_min, v_max):
    return max(v_min, min(v_max, value))


class JpegParser:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.trace_log = io.StringIO()

    def trace(self, *args, **kwargs):
        if self.verbose:
            trace(*args, **kwargs)
        print(*args, file=self.trace_log, **kwargs)

    @property
    def parse_log(self):
        return self.trace_log.getvalue()

    def load(self, fp):
        image = JpegImage()

        self.trace(': reading segments')

        for s in JfifReader(fp):
            seg = self._parse_segment(s)
            if not seg:
                self.trace(s)
                image.unprocessed.append(s)
            else:
                self.trace(seg)
                self._process_image_segment(image, seg)

        image.size = (image.frame.x, image.frame.y)

        self.trace(': decoding scan')
        image.data_quantized = self._decode_scan(image.frame, image.scan, image.huffman_tables,
            restart=image.restart_interval)

        self.trace(': dequantizing')
        image.data_dequantized = self._dequantize(image.scan, image.data_quantized, image.quantization_tables)

        self.trace(': inverting DCT')
        image.samples = self._inverse_transform(image.frame, image.data_dequantized)

        self.trace(': reading YCbCr')
        image.ycbcr = self._expand_mcus(image.scan, image.size, image.samples)

        self.trace(': reading RGB')
        image.rgb = self._convert_to_rgb(image.frame, image.ycbcr)

        return image

    def _parse_segment(self, segment):
        if segment.marker in (0xFFD8, 0xFFD9, 0xFFE0):
            return segment

        elif segment.marker == 0xFFC4:
            return JpegHuffmanTables.frombytes(segment.data)

        elif segment.marker == 0xFFDA:
            return JpegStartOfScan.frombytes(segment.data)

        elif segment.marker == 0xFFDB:
            return JpegQuantizationTables.frombytes(segment.data)

        elif segment.marker == 0xFFDD:
            return JpegRestartInterval.frombytes(segment.data)

        elif ((segment.marker & 0xFFF0) == 0xFFC0) and ((segment.marker & 0xF) in (0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15)):
            return JpegStartOfFrame.frombytes(segment.data, marker=segment.marker)

    def _process_image_segment(self, image, seg):
        if isinstance(seg, (JpegStartOfImage, JpegEndOfImage)):
            pass

        elif isinstance(seg, JfifApp0):
            image.jfif = seg

        elif isinstance(seg, JpegRestartInterval):
            image.restart_interval = seg.size

        elif isinstance(seg, JpegQuantizationTables):
            tables = image.quantization_tables or list()
            tables.extend(seg.tables)
            image.quantization_tables = tables

        elif isinstance(seg, JpegHuffmanTables):
            tables = image.huffman_tables or list()
            tables.extend(seg.tables)
            image.huffman_tables = tables

        elif isinstance(seg, JpegStartOfFrame):
            image.frame = seg

        elif isinstance(seg, JpegStartOfScan):
            image.scan = seg

    def _decode_scan(self, frame, scan, huffman_tables, restart=None):
        huff = dict(((t.destination, t.kind), HuffmanCodec(t.data)) for t in huffman_tables)
        chuf = {(cid, k):huff[([dct, act][k], k)] for (cid, dct, act) in scan.components for k in (0, 1)}
        su, sv = max(u for (cid, u, v, _) in frame.spec), max(v for (cid, u, v, _) in frame.spec)
        sampling_factor = su * sv
        w = int(math.ceil(frame.x / 8 / su) * su)
        h = int(math.ceil(frame.y / 8 / sv) * sv)

        def decode_segment(data, total):
            stream = BitStream(data)
            n = len(frame.components_iter)
            res = list()
            prev = [0] * n
            for i in range(total):
                mcu = [[self._decode_component(stream, cid, chuf[(cid, 0)], chuf[(cid, 1)]) for _ in range(u * v)]
                    for (cid, u, v, _) in frame.spec]

                lin = [x for c in mcu for x in c]
                for j in range(n):
                    prev[j] = lin[j][0] = prev[j] + lin[j][0]

                res.append(mcu)
            return res

        if restart:
            segments = scan.data.split(b'\xFF\xD0')
            mcus = [mcu for seg in segments for mcu in decode_segment(seg, restart)]
        else:
            mcus = decode_segment(scan.data, w * h // sampling_factor)

        grid = dict()
        for i, mcu in enumerate(mcus):
            for j, sample in enumerate(itertools.product(*mcu)):
                off = (i%(w//su)*su+j%su, i//(w//su)*sv+j//su)
                grid[off] = sample

        return [grid[(x,y)] for y in range(h) for x in range(w)]

    def _decode_component(self, stream, cid, dc_table, ac_table):
        def category_value(s, v):
            m = 2 ** s
            if v < m // 2:
                v -= m - 1
            return v

        # F.1.2.1 Huffman encoding of DC coefficients
        # F.2.2.1 Huffman decoding of DC coefficients
        s = dc_table.feed(stream)
        dc = stream.unpack(s)
        dc = category_value(s, dc)

        # F.1.2.2 Huffman encoding of AC coefficients
        # F.2.2.2 Decoding procedure for AC coefficients
        ac = list()
        while len(ac) < 63:
            rs = ac_table.feed(stream)
            r, s = (rs >> 4), (rs & 0xf)
            if r == 0 == s:
                ac.extend([0] * (63 - len(ac)))
            else:
                ac.extend([0] * r)
                v = stream.unpack(s)
                v = category_value(s, v)
                ac.append(v)
        assert len(ac) == 63, len(ac)

        return [dc] + ac

    def _dequantize(self, scan, quantized, tables):
        def deq(dct, unit):
            table = tables[dct]
            return [(x * q) for x, q in zip(unit, table.data)]

        return [[deq(dct, unit) for dct,unit in zip((dct for _,dct,_ in scan.components), mcu)]
            for mcu in quantized]

    def _inverse_transform(self, frame, dequantized):
        level_shift = 2 ** (frame.precision - 1)
        sample_precision = (2 ** frame.precision) - 1
        idct_table = self._idct_table

        if np:
            units = np.array(dequantized, dtype=np.float64)
            table = np.array(idct_table, dtype=np.float64)
            samples = np.matmul(units, table.T)
            samples = np.clip(np.around(samples) + level_shift, 0, sample_precision)
            return samples.astype(np.uint16).tolist()

        else:
            def idct(unit):
                def idctxy(xy):
                    table = idct_table[xy]
                    r = sum(x * q for x, q in zip(unit, table))
                    return _clamp(round(r) + level_shift, 0, sample_precision)
                return [*map(idctxy, range(64))]

            return [[idct(unit) for unit in mcu] for mcu in dequantized]

    _zigzag8 = [
        [ 0,  1,  5,  6, 14, 15, 27, 28],
        [ 2,  4,  7, 13, 16, 26, 29, 42],
        [ 3,  8, 12, 17, 25, 30, 41, 43],
        [ 9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63],
    ]

    def _generate_idct_table(zigzag):
        r2sq = 1 / dec(2).sqrt()
        pi = dec(math.pi)
        def cos(x): return dec(math.cos(x))

        irev = [p for _,p in sorted((col,(y,x)) for y,row in enumerate(zigzag) for x,col in enumerate(row))]

        def idct(x, y):
            x, y = map(dec, (x, y))
            t = [(((1 if u else r2sq) * (1 if v else r2sq)
                * cos((2 * x + 1) * u * pi / 16)
                * cos((2 * y + 1) * v * pi / 16)
                ) / 4)
                for u in range(8) for v in range(8)]
            return [float(t[v*8+u]) for v,u in irev]

        return [idct(x, y) for x in range(8) for y in range(8)]

    _idct_table = _generate_idct_table(_zigzag8)

    def _expand_mcus(self, scan, image_size, samples):
        w, h = image_size
        mw = (w + 7) // 8
        return [[samples[y//8*mw+x//8][i][y%8*8+x%8] for i,_ in enumerate(scan.components)]
            for y in range(h) for x in range(w)]

    def _convert_to_rgb(self, frame, samples):
        level_shift = 2 ** (frame.precision - 1)
        sample_precision = (2 ** frame.precision) - 1

        def decode(sample):
            y, *chrom = sample
            if not chrom:
                return (y, y, y)
            cb, cr = chrom
            r = _clamp(round(y + 1.402 * (cr - level_shift)), 0, sample_precision)
            g = _clamp(round(y - (0.114 * 1.772 / 0.587) * (cb - level_shift) - (0.299 * 1.402 / 0.587) * (cr - level_shift)), 0, sample_precision)
            b = _clamp(round(y + 1.772 * (cb - level_shift)), 0, sample_precision)
            return (r, g, b)

        return [*map(decode, samples)]


class ImageWriter:
    def export(self, image, file_name):
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()
        if ext == '.ppm':
            self._export_ppm(image, file_name)
        else:
            raise JpegReaderError(f'Not supported image format: {ext}')

    def _export_ppm(self, image, file_name):
        with open(file_name, 'wb') as so:
            so.write(b'P6\n')
            so.write('{} {}\n'.format(*image.size).encode('ascii'))
            so.write(b'255\n')
            so.write(bytes(x for p in image.rgb for x in p))


def dump_image(image, parser, todir=None):
    if todir:
        os.makedirs(todir, exist_ok=True)
    else:
        todir = '.'

    with open(os.path.join(todir, 'parse-log.txt'), 'w') as f:
        f.write(parser.parse_log)

    with open(os.path.join(todir, 'data-quantized.pickle'), 'wb') as f:
        pickle.dump(image.data_quantized, f)

    with open(os.path.join(todir, 'data-dequantized.pickle'), 'wb') as f:
        pickle.dump(image.data_dequantized, f)

    with open(os.path.join(todir, 'samples.pickle'), 'wb') as f:
        pickle.dump(image.samples, f)

    with open(os.path.join(todir, 'pixels-ycbcr'), 'wb') as f:
        f.write(bytes(x for mcu in image.ycbcr for x in mcu))

    with open(os.path.join(todir, 'pixels-rgb'), 'wb') as f:
        f.write(bytes(x for mcu in image.rgb for x in mcu))


def parse_jpeg(fp, dump_dir=None, export=None, verbose=False):
    parser = JpegParser(verbose=verbose)
    image = parser.load(fp)

    if export:
        ImageWriter().export(image, export)

    if dump_dir:
        dump_image(image, parser, todir=dump_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=argparse.FileType('rb'), help='JPEG file to parse')
    parser.add_argument('-d', '--dump', metavar='dir', type=str, help='Dump directory')
    parser.add_argument('-o', '--export-image', metavar='ofile', type=str, help='Image file to export: .ppm')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    parse_jpeg(
        args.infile,
        dump_dir=args.dump,
        export=args.export_image,
        verbose=args.verbose,
    )
