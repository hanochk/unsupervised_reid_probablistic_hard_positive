#!/usr/bin/env python3
import io
import zlib

def stream_unzipped_bytes(filename):
    """
    Generator function, reads gzip file `filename` and yields
    uncompressed bytes.

    This function answers your original question, how to read the file,
    but its output is a generator of bytes so there's another function
    below to stream these bytes as text, one line at a time.
    """
    with open(filename, 'rb') as f:
        wbits = zlib.MAX_WBITS | 16  # 16 requires gzip header/trailer
        decompressor = zlib.decompressobj(wbits)
        fbytes = f.read(16384)
        while fbytes:
            yield decompressor.decompress(decompressor.unconsumed_tail + fbytes)
            fbytes = f.read(16384)


def stream_text_lines(gen):
    """
    Generator wrapper function, `gen` is a bytes generator.
    Yields one line of text at a time.
    """
    try:
        buf = next(gen)
        while buf:
            lines = buf.splitlines(keepends=True)
            # yield all but the last line, because this may still be incomplete
            # and waiting for more data from gen
            for line in lines[:-1]:
                yield line.decode()
            # set buf to end of prior data, plus next from the generator.
            # do this in two separate calls in case gen is done iterating,
            # so the last output is not lost.
            buf = lines[-1]
            buf += next(gen)
    except StopIteration:
        # yield the final data
        if buf:
            yield buf.decode()


# Sample usage, using the stream_text_lines generator to stream
# one line of RDF text at a time
zip_file = False
if zip_file:
    bytes_generator = (x for x in stream_unzipped_bytes('freebase-rdf-latest.gz'))
else:
    bytes_generator = (x for x in open('/media/celeb1m/freebase-rdf-latest', "rb"))
for line in stream_text_lines(bytes_generator):
    # do something with `line` of text
    print(line, end='')
    break

"""
https://stackoverflow.com/questions/51244854/extract-data-dump-from-freebase-in-python
Share
Follow
edited Jul 12, 2018 at 13:40
answered Jul 12, 2018 at 12:28
Ryder Lewis's user avatar
Ryder Lewis
176
"""
