Zillow 2017 Predictions

Project Description
 - Utilize regression models to predict home values for single family homes
    - Find key features that drive Tax Appraised Values (target)
    - Beat original Zillow model(baseline)

Initial Hypothesis
 - Tax appraised value is driven by square feet, lot size, and number of bedrooms and bathrooms
 - County will impact the tax appraised value of the homes

Data Dictionary
 - taxvaluedollarcnt(Tax_Appraised_Value - target)
 - bedroomcnt(Number_of_Bedrooms)
 - bathroomcnt(Number_of_Bathrooms)
 - calculatedfinishedsquarefeet(Square_Feet)
 - lotsizesquarefeet(Lot_Size)
 - yearbuilt(Year_Built)
 - fips(County) - LA County, Orange County, Ventura County
 - taxamount(Tax_Assessed)

 Plan
  - Hypothesize initial questions
  - Determine deliverables
  - Construct clear goals 
 Acquire
  - Import zillow data from Codeup database
 Prepare
  - Clean and prep data by dropping nulls, converting county codes to actual county names
  - Scale data utilizing MinMax scaler
  - Split data into train, validate, test datasets
 Explore
  - Explore the interactions of features and target variable to determine drivers of tax appraised value.
  - Provide at least four visualizations and two statistical tests.
 Model
  - Utilize regression models to predict tax appraised values for zillow dataset
  - Show the three best models
  - Test the best performing model (determined by results of train and validate datasets)

Reproduce this Project
- Tn order to reproduce this project you will need your own .env file with Codeup database credentials, as well as:
jupyter notebook
wrangle.py
Zillow - FINAL.ipynb

Key Findings, Recommendations, Takeaways
 - Square Feet is the most signifcant feature in determining tax appraised value
 - The polynomial model is the best regression model to predict tax appraised value
 - Utilizing a polynomial model on Ventura County alone produces the best prediction
 - I recommend gathering data for school ratings, walkability scores, and neighborhood activities for better predictions
 - With more time, I'd like to explore the relationship between zipcodes and tax appraised values