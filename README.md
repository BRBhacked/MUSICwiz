# 🎧 MusicWiz

**MusicWiz** is a machine learning-powered music recommendation system built during a hackathon by a team of three. It leverages Spotify’s Web API and unsupervised learning techniques to demonstrate how intelligent music recommendations can be generated using user behavior and track features.

---

## 🚀 Getting Started

These instructions will help you set up and run the project locally for development and testing.

### 🔧 Prerequisites

Ensure you have Python installed along with the following libraries:

- `spotipy`
- `pandas`
- `numpy`
- `scikit-learn`

You can install them using pip:

```bash
pip install spotipy pandas numpy scikit-learn
````

Additionally, register an app at the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/) to obtain the following credentials:

* `client_id`
* `client_secret`

These are required to authenticate API requests to Spotify.

---

## 📦 Installation

Clone this repository:

```bash
git clone https://github.com/your-username/music-recommender-system.git
cd music-recommender-system
```

---

## ▶️ Running the Code

Run the main script:

```bash
python music_recommender.py
```

Follow the prompts to authenticate via Spotify, and the system will begin analyzing your listening data to recommend similar tracks based on feature clustering.

---

## 🛠 Built With

* [Spotipy](https://spotipy.readthedocs.io/) – Python client for the Spotify Web API
* [Pandas](https://pandas.pydata.org/) – For data manipulation
* [NumPy](https://numpy.org/) – For numerical operations
* [Scikit-learn](https://scikit-learn.org/) – For clustering and machine learning

---

## 👨‍💻 Authors

* [Aryaman](https://github.com/araina25)
* [Agrim](https://github.com/aggi000)
* [Falak](https://github.com/fa1ak)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE.md).

---

## Acknowledgments

* Thanks to **Spotify** for providing access to their API and datasets.
* Big appreciation to the **hackathon organizers** for the platform to build and innovate.

