# Udacity-Data-Scientist-Nanodegree-Projects
## Projects completed within the Udacity Data Scientist Nanodegree Program

### Disaster Response Pipeline Project

When natural disasters hit communities, there is often a severe lack of organized communication between those in need and the agencies trying to assist these people. While each request for assistance from individuals and groups within these communities are typically very diverse, there are larger buckets that we can categorize these requests into to assist relief agencies in efficiently working with these communities. Thus, creating a more systematic, organizational categorization of messages to aide in this effort would be a huge benefit.

In this project, I worked on a dataset containing messages related to disaster relief efforts along with corresponding categories of each message. Using the messages and associated categories, I built a ML pipeline (and simple web app) to categorize each event/message in order to appropriately send the message to the necessary disaster relief agency.

#### Files:

*process_data.py*:

Loads up the message and categorical data from local csv files. These files are then combined into a dataframe that is subsequently cleaned. This df/table is then uploaded into a SQLite database.

*train_classifier.py*:

Loads table from the database, creates a ML pipeline, and uses GridSearchCV to tune the model hyperparameters. Subsequent fitting and prediction using the "optimal" model is performed. The model is then saved to a pickle file to be run on the web app.

*run.py*:

Runs a simple web app that displays two bar charts detailing genre and category distribution within the data set on the home page. Then, when a message is input into the visual classifier, the model predicts which category the message should belong to per the images of the webpage below.

![Web App](/images/Web%20app1.png)

![Web App](/images/Web%20app2.png)
