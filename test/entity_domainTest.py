from app.pipeline.metadata import MetadataExtractor

print("Initializing Extractor...")
extractor = MetadataExtractor()

text = "Jakob Bernoulli introduced the Law of Large Numbers in 1713."
print(f"\nAnalying: '{text}'")

meta = extractor.extract_metadata(text)

print("\n--- Results ---")
print(f"Entities: {meta.entities}")
print(f"Relations: {meta.relations}")
print(f"Domain: {meta.domain}")