
import pandas as pd

df = pd.read_csv("C:/Users/pc/PycharmProjects/video-captioning/Frame_FPS.csv")
sorted_df = df.sort_values(by=["Frames"])
sorted_df.to_csv('Frames_and_FPS_of_videos.csv', index=False)