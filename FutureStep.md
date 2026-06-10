 - implement a method to
   - stack all of your detected images with a shared center (just calculate the moment and transform them together)
   - crop to the largest rectangle possible
   - Stack reg all of these together


 - What pipeline I should follow
   - Take video
   - reformat to SD
   - Run a model to take out the sclera (we can use CV for this, but is slow rn)
   - mask out that area so we only have the sclera
   - Let the user specify a region of frames they like
   - stack em with same moment
   - let user define region to zoom/crop
   - stack reg all of them
   - Then isolate vessels