# Machine learning to identify phospholipidosis inducers


# The code employs the Random Forest (RF) algorithm to predict the phospholipidosis of small molecules. To this end, the molecular representations consist of the MACCS fingerprint combined with two physicochemical properties, namely the pKa and cLogP. Prior to training the model, the cLogP and pKa values are converted into binary features. Specifically, a bit is set to one if the corresponding value is greater than or equal to 3.0 (cLogP) or 7.4 (pKa), and set to zero otherwise. This approach is adopted to make the continuous properties compatible with the binary nature of the fingerprints.


# Here, we attached our curated PL dataset from literature.
