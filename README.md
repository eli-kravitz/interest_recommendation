# interest_recommendation

This repo contains code to learn and infer human interest in infrared target tracks across the globe. The model that was used to learn interest is very similar to a vanilla Bayesian logistic regression, but with the added complexity of dependencies between input variables and partially observable input variables. The model is shown below:

<img src="https://your-image-url.type" width="100">

The main backend code exists in `backend/algorithms/sandbox_ek/interesting_track_GM2/camp_interest_classifier.py` and `backend/algorithms/sandbox_ek/interesting_track_GM2/camp_helper_functions.py`, with the other files mainly existing for test purposes.

See the file `EliKravitz_Thesis_Final_sent.pdf` for a full explanation of the model and its validation.
