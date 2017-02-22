#**Finding Lane Lines on the Road** 

## Pipeline

The detection of the lane marking was done in 6 general steps:
* Converting image to greyscale
* Applying gausian filter on the image 
(Remove noise from image which helps us to get cleaner results from next step)
* Perform Canny edge detection
(Detects edges in the image) 
* Apply mask to canny edge image (We know that lane lines are in some region infront of the car and don't want to process other edges)
* Detecting lines using Hough transformation
(Detects lines from set of points)
* Connecting hough lines into lane lines


Step 6 is new and wasn't used in the lesson previously

## Connecting the lines
Our goal in this project was to get connected lane lines. My solution was to get this lines using linear regression.
The detection of the lane lines was done in three steps:
* Calculating slope for each line and removing all lines which absolute slope value is greater then 0.5 
* Splitting lines on left and right based on their slope sign and position on the screen
* Use linear regression to get left and right lane lines

Then we can draw lanes based on their slope and b parameters and some length (for simplicity used hardcoded value)


## Shortcomings
What worked quite well on test images and white and yellow videos didn't work so well on challendge one:
* Algorithm wasn't able to find lane lines on light patch spot on the road 
* Additional noise from the tree shadows
* Car was turning and lane line wasn't straight any more

In addition we would have other problems with current pipeline:
* Masking region is hardcoded and it would work bad if the car not in the middle of the lane
* Datecting lanes while switch road lanes would be also a problem
* Changing light condition and road color would make algorithm work unstable


## Improvements

To improve performance on challenge video was made next steps:
* Greysacale transformation was removed from pipeline (this allowed to get canny edges even on light road patch)
* Was tuned some canny edge and hough transformation parameters 
* Was added line smoothing (Outputed line is avarage of current and some previous lines. This helpes with line 'jumps' a lot.  Potentially dengerous in sharp turnes, but we can use sensors to understand when we should switch it off or use smaller smooth window)

Other potential improvements:
* Either detect canny edge and hough parameters based on light and road conditions or run algorithm with different parameters several times trying to get best results. 
* Improving detecting outliers while generating lane lines. (Maybe run linear regression twice, removing outliers after first run)
