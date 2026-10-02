# Synthetic decoder fixture

`jpeg-extraneous-byte.jpg` is a 16×16 solid RGB `(180, 60, 40)` image, encoded with `cjpeg` from a binary PPM. A single byte `0x42` was inserted immediately before its `FF D9` end marker. It reproduces TurboJPEG's `Corrupt JPEG data: 1 extraneous bytes before marker 0xd9` diagnostic without using a user's photograph.

The regression checks that this strict decoder error skips only that image, reports its path and reason, and allows later images to be indexed. No external encoder is needed to run the tests.
