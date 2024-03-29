# Machine learning to identify chemical probes that induce phospholipidosis (PL)
![image](https://github.com/HuabinHu/ML-for-PL-prediction/assets/115711932/af868e0f-11df-4b67-aa65-f4c356ea494b)



1. The code employs the Random Forest (RF) algorithm to predict the phospholipidosis of small molecules. 


2. To this end, the molecular representations consist of the MACCS fingerprint combined with two physicochemical properties, namely pKa and cLogP. Prior to training the model, the cLogP and pKa values are converted into binary features. Specifically, a bit is set to one if the corresponding value is greater than or equal to 3.0 (cLogP) or 7.4 (pKa), and set to zero otherwise. This approach is adopted to make the continuous properties compatible with the binary nature of the fingerprints.


3. Here, we attached our curated PL dataset from literature. In the file, the column name with "Class" indicated PL annotations reported in literature with "1" representing PL inducer and "0" non-inducer.
