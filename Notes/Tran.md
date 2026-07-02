## Reading Notes: The Normalized Compression Distance and Image Distinguishability
### Tran (Proceedings of SPIE, Human Vision and Electronic Imaging XII, 2007)

> **Summary:** This paper asks whether a math formula built from file compression can predict if two images look same to a person. The formula is called Normalized Compression Distance, or NCD. It comes from an idea called Kolmogorov complexity, which measures how short a computer program would need to be to produce a given piece of data. Since that idea cannot actually be computed, author swaps in real compression programs like gzip and bzip2 as stand ins. The author repeats a 1936 study by psychologist E. Goldmeier, where people picked which of two changed drawings looked more like an original drawing. He then compares NCD's picks to what human subjects picked. The NCD does well when images differ by simple additions or removals, but does poorly when differences involve shape, material, or arrangement of parts. Finally, author tests NCD in digital watermarking, where two images must look identical to a person even though their pixel data differs. Here NCD is unreliable. A rotated image looks completely different to human eye but scores as very similar under NCD, while a defaced copy of an image scores as more similar to original than a correctly watermarked copy does. The author proposes a new definition of visual sameness based on size of pixel by pixel difference, and closes by noting this measure still needs more work.

---

### Works Cited

* Tran, Nicholas. "The Normalized Compression Distance and Image Distinguishability." *Proceedings of SPIE, Vol. 6492: Human Vision and Electronic Imaging XII*, 64921D, 2007. https://doi.org/10.1117/12.704334