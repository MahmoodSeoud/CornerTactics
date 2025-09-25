from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="../sn-gamestate/data/SoccerNetGS")
mySoccerNetDownloader.downloadDataTask(task="gamestate-2024", split=["train", "valid", "test", "challenge"])
