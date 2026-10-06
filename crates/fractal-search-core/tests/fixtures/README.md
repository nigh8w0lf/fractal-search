# Synthetic decoder fixture

`jpeg-extraneous-byte.jpg.hex` stores whitespace-separated hexadecimal bytes for a 16×16 solid RGB `(180, 60, 40)` image, encoded with `cjpeg` from a binary PPM. A single byte `0x42` was inserted immediately before its `FF D9` end marker. It reproduces TurboJPEG's `Corrupt JPEG data: 1 extraneous bytes before marker 0xd9` diagnostic without using a user's photograph.

The regression checks that this strict decoder error skips only that image, reports its path and reason, and allows later images to be indexed. No external encoder is needed to run the tests.

Tests decode the text directly into their temporary catalogs. The reconstructed JPEG is byte-for-byte identical to the original 634-byte fixture; no image file or external encoder is needed in the checkout. Invalid hex fails the test explicitly.
