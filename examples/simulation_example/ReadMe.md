# Simulation example

I want to do a few things here.

1. Test out current kinematics library on a real problem, update library as required.
2. I'd like to test out an idea from a YT video
3. I'd like to get better at tracking and data fusion as a thing
4. I'd like to get better at working with camera's as sensors (or at least setting conditions for this)

## YT video

There's a YT video below about tracking stealth fighter jets with cheap cameras. The idea is to have heaps of them networked together. The cameras report all targets (detected moving pixels) which will return hits from birds, flies, planes, helicopters, everything that moves.

We can convert all these into an inertial frame and draw a unit vector from the camera, the target must exist somewhere on that line. Trouble is we don't know where. By finding the points the intersection points we can estimate inertial position, and start a track for each target.

On the next frame we can do the same thing, marry up detections to track and back calculate the required movement for that detection to occur. Any close object should move in a way that it's physically impossible for it to be a true detection, however distant objects like aircraft should move 'sensibly' in inertial space. At each point tracks will be pruned and probable tracks will be put forward.

The assertion of the video is that you could cover a wide area with a reasonable number of cameras.

### I'd like to

- Simulate this to see it work, it's a neat idea but I'd like to see it / do it
- Find out how what the requirements on the camera's would be (number, resolution .... > cost?) just to see if the idea is workable in practice

### Implementation

- Let's do a quick prototype in 2d
- Camera's provide bearings only with 1 true target an a random number of extra false bearings

Tracker process:
1. Calculate intersection points for each direction vector from each combination of returns (cross product)
2. Start a tracker for each point in inertial with CV
3. Run a nearest neighbour track
4. Produce animation


## References

- https://www.youtube.com/watch?v=m-b51C82-UE
