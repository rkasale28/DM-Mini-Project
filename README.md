# Youtube Data Analyser

#### Designed and developed a web application for analysing and classifying Youtube dataset.

> YouTube is an online video-sharing platform which helps the content creators reach out to audience and present their talent. This helps them earn the revenue. Therefore, it becomes important that the video of a particular category is recommended to the user, who is interested in that category. This will help the user in finding relevant videos. Also, this will help the content creators increase their subscribers. To increase the number of likes on a video, views play a crucial role and a greater number of views in turn will result in more subscriber count. Therefore, we aim to find out how the videos are served to a particular user using data mining strategies and how the number of views is related to the number of likes

YouTube videos incorporate a number of different features such as likes, views, comments, titles, etc. Based on relations between these features YouTube segregates the videos into categories. In this project, we apply data mining strategies to find out how YouTube determines the category of a particular video, so that the particular category could be served as a recommendation to the pertinent user. For this purpose, we use Naïve Bayes Classification Technique, KNN classification and Simple Linear Regression.


**Important Features:**
* The dataset used represented the day to day list of trending videos in India. It was collected using the Youtube API.
* This dataset was analysed for predicting category of a particular video.
* This project uses Django Framework.
* This project Matplotlib for plotting the results.
* The dataset includes several days of record with up to 5000 videos. Data includes the attributes as follows:
  * title
  * channel_title
  * publish_time
  * trending_date
  * views
  * likes
  * dislikes
  * comment_count
  * comments_disabled
  * ratings_disabled
  * category_id
* The category_id is an integer representing various categories. The category and their category_id is specified as follows:

| Category ID | Category |
| :---:    | :------: |
| 22   | People and Blogs |
| 23   | Comedy |
| 24   | Entertainment |
| 25   | News and Politics |
| 26   | How-to |
| 27   | Education |
| 28   | Science and Technology |

* Naive Bayes Algorithm was used for predicting the category.
* Similarly, KNN clustering algorithm was used.
* Also, linear regression was used for analysing the relation between views and likes on a particular video.

**Please Note:** The project was developed for the demonstration of working of these algorithms. Therefore, we don't use any inbuilt package. Rather, we have constructed our own functions for demonstration of these algorithms.
