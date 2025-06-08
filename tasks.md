## Tasks

- [ ] Switch off / Switch on to work - the only thing required is switching on, everything must boot and qt screen should be dislayed 

- [ ] (optional) hibernation mode -  closing off the xiomi camera should hibernate the cv2 recongnition pipeline. 2 seconds check to see if the camera is open to come back from hibernation 

- [ ] RECOGNITION: zero false-positive recognition pipeline, with <200ms 

- [ ] Proper sync with neon-singapore-db on every recognition through flask server 

- [ ] react native - mobile app to show the stats to dad (closest to realtime) 

- [ ] web version of attendance stats

- [x] db schema for neon for attendance 

- [ ] db schema for local sqlite database for axon

- [ ] flask APIs for talking to backend 

- [ ] backend APIs to talk to flask 

- [ ] proper caching layer to restrict detection to a single instance per session

- [ ] separation of concerns / proper structure for the source code of attendance system 

- [ ] enrollment pipeline through flask 

    - in `enrollment_images` => image 
    - in sqlite database => admissionNumber, room, name
    - extract features and store them in faiss_index

