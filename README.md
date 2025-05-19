# WiDS Datathon 2025
Participants are tasked with building a model to predict both an individual’s sex and their ADHD diagnosis using **functional brain imaging data** of children and adolescents and their socio-demographic, emotions, and parenting information. ADHD occurs in about 11% of adolescents, with around 14% of boys and 8% of girls having a diagnosis. Females are much harder to predict and diagnose.

Data includes:
+ The training folder `train_new_tsv` consisting of 3 types of information about the 1200+ subjects. They are:
	+ The targets (ADHD diagnosis and sex)
	+ Functional MRI connectome matrices
	+ Socio-demographic information e.g., subject’s “handedness” or “parent’s education level”, emotions (“Strength and Difficulties Questionnaire”), and parenting information (“Alabama Parenting Questionnaire”). These include both quantitative and categorical metadata.
+ The test folder `test_tsv` consists of unseen data frames for 300+ subjects. Metadata is the same for training set. 

-----
# Feature Importance

AHDH Outcome

![Default](attachments\adhd_features.png "AHDH Feature Importance")

Sex

![alt text](attachments\sex_features.png "AHDH Feature Importance")

