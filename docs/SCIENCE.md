# The science behind AffectLab

AffectLab is built around one idea from affective science: a face is not a
readout of an emotion. It is a set of muscle movements, produced in a
context, by a body with a heartbeat and a breathing rhythm. So instead of
printing a single word above a face, AffectLab measures several layers and
keeps them separate, so that each can be judged on its own evidence.

This document explains each layer, the method used, and where it comes from.
Every number that matters carries a citation; the reference list is at the
end. The companion file [ETHICS.md](ETHICS.md) covers what these
measurements do not mean.

## 1. Why "emotion detection" is the wrong frame

The most thorough review of the field to date examined more than a thousand
studies and concluded that the common view (a scowl means anger, a smile
means happiness) is not supported: the same emotion is expressed in many
ways, the same facial movement occurs in many emotional states, and both
vary across people, situations and cultures [1]. Facial movements carry
*some* information about affect, but far less than the labels suggest, and
context matters as much as the face.

Two consequences shape AffectLab's design:

1. **Describe before you interpret.** The Facial Action Coding System (FACS)
   describes what the face is doing without claiming what the person feels
   [2, 3]. AffectLab reports Action Units first, and treats categorical
   labels as one interpretation layered on top.
2. **Prefer dimensions to categories.** Dimensional models place affect on
   continuous axes of valence and arousal rather than in discrete boxes
   [4, 5]. A continuous signal can be smoothed, tracked and analysed for
   dynamics; a label can only flicker.

The physiological layer comes from the tradition of affective computing
[6], where facial, vocal and bodily signals are combined because no single
channel is reliable on its own.

## 2. Facial geometry: landmarks and blendshapes

MediaPipe's Face Landmarker produces 478 three-dimensional landmarks per
face from a single RGB frame, using a lightweight neural mesh regressor with
an attention refinement stage around the eyes and lips [7, 8]. Alongside
the mesh it predicts 52 *blendshape* coefficients in [0, 1], the same set
Apple's ARKit uses to drive animated avatars, and a 4x4 facial
transformation matrix from which head orientation is read directly.

AffectLab uses the landmarks for geometry (eye aspect ratio, skin regions,
head pose) and the blendshapes for Action Units.

## 3. Action Units (FACS)

FACS [2, 3] decomposes facial behaviour into Action Units, each tied to the
contraction of a specific muscle or muscle group. AU12, "lip corner puller",
is the zygomaticus major; AU4, "brow lowerer", is the corrugator group; AU6,
"cheek raiser", is the orbicularis oculi. FACS coders are certified after
roughly a hundred hours of training, and automatic AU estimation is an
active research field.

The ARKit blendshape set is itself FACS-inspired, so a transparent table
maps blendshapes to AUs. AffectLab averages bilateral pairs and clamps the
result to [0, 1]:

| AU | Name | Blendshapes | Notes |
|---|---|---|---|
| AU1 | Inner brow raiser | browInnerUp | |
| AU2 | Outer brow raiser | browOuterUpLeft, browOuterUpRight | |
| AU4 | Brow lowerer | browDownLeft, browDownRight | |
| AU5 | Upper lid raiser | eyeWideLeft, eyeWideRight | |
| AU6 | Cheek raiser | cheekSquintLeft, cheekSquintRight | Duchenne marker; often weak in blendshapes |
| AU7 | Lid tightener | eyeSquintLeft, eyeSquintRight | |
| AU9 | Nose wrinkler | noseSneerLeft, noseSneerRight | |
| AU10 | Upper lip raiser | mouthUpperUpLeft, mouthUpperUpRight | |
| AU12 | Lip corner puller | mouthSmileLeft, mouthSmileRight | |
| AU14 | Dimpler | mouthDimpleLeft, mouthDimpleRight | |
| AU15 | Lip corner depressor | mouthFrownLeft, mouthFrownRight | |
| AU16 | Lower lip depressor | mouthLowerDownLeft, mouthLowerDownRight | |
| AU17 | Chin raiser | mouthShrugLower | |
| AU18 | Lip pucker | mouthPucker | |
| AU20 | Lip stretcher | mouthStretchLeft, mouthStretchRight | |
| AU22 | Lip funneler | mouthFunnel | |
| AU23 | Lip tightener | mouthPressLeft, mouthPressRight | proxy shared with AU24 |
| AU24 | Lip pressor | mouthPressLeft, mouthPressRight | |
| AU26 | Jaw drop | jawOpen | |
| AU28 | Lip suck | mouthRollLower, mouthRollUpper | |
| AU29 | Jaw thrust | jawForward | |
| AU34 | Cheek puff | cheekPuff | |
| AU43 | Eyes closed | eyeBlinkLeft, eyeBlinkRight | AU45 (blink) is the temporal event |

This is an approximation. Blendshapes are trained to drive avatars, not to
match human FACS coding, and their intensities are not calibrated against
FACS intensity scores (A to E). Treat the AU values as a consistent,
repeatable description of movement rather than a certified FACS code.

### The rule-based emotion backend

EMFACS [9] and the FACS Investigator's Guide [3] list AU combinations
typically associated with prototypical expressions. AffectLab's `facs`
backend scores each prototype as a weighted mean of its AU intensities:

| Expression | Supporting Action Units (weight) | Inhibited by |
|---|---|---|
| happiness | AU6 (0.5), AU12 (1.0) | AU15 |
| sadness | AU1 (1.0), AU4 (0.6), AU15 (1.0), AU17 (0.4) | AU12 |
| surprise | AU1 (0.7), AU2 (0.7), AU5 (1.0), AU26 (1.0) | AU4, AU20 |
| fear | AU1 (0.6), AU2 (0.5), AU4 (0.6), AU5 (1.0), AU7 (0.4), AU20 (1.0), AU26 (0.5) | AU12 |
| anger | AU4 (1.0), AU5 (0.6), AU7 (0.8), AU23 (1.0) | AU1, AU12 |
| disgust | AU9 (1.0), AU10 (0.6), AU15 (0.5), AU16 (0.4), AU17 (0.3) | AU12 |
| contempt | unilateral AU12 or AU14 (asymmetry) | a symmetric smile |
| neutral | one minus the strongest prototype score | |

The support score is the weighted mean of the supporting AUs (with a
blendshape intensity of 0.7 counted as fully active). Inhibitory AUs then
scale it down: a lowered brow or stretched lips argue against surprise and
for fear, which is exactly how EMFACS separates the two, and a smile argues
against sadness, anger and disgust.

Scores are sharpened (squared) and normalised. The backend returns the AU
intensities that drove its decision as `evidence`, so every label is
explainable. It has no learned parameters and no training data, which is
both its strength (transparency) and its weakness (it only knows
prototypes).

## 4. Categorical emotion from a neural network

The `ferplus` backend runs the FER+ convolutional network [10]. FER+
re-annotated the FER2013 faces with ten crowd workers each and trained on
the resulting label *distributions*, which is a more honest target than a
single majority label: many faces are genuinely ambiguous. It outputs eight
classes (neutral, happiness, surprise, sadness, anger, disgust, fear,
contempt), which AffectLab adopts as its canonical set. Input is a 64x64
greyscale crop. The ONNX export runs in a few milliseconds on a CPU.

The `deepface` backend keeps the original project's DeepFace pipeline [11]
available for comparison. It is much heavier (TensorFlow).

The default `ensemble` averages `ferplus` (weight 0.6) and `facs` (weight
0.4). The two make different mistakes: the network reacts to texture and
lighting, the rules react to geometry. Averaging reduces label flicker and
keeps the AU evidence attached to every estimate.

## 5. Dimensional affect and emotion dynamics

Russell's circumplex model [4] arranges affective states on two orthogonal
axes: valence (unpleasant to pleasant) and arousal (deactivated to
activated). Posner, Russell and Peterson [5] argue that this two-system
account fits neurophysiological evidence better than basic-emotion theories.

AffectLab places each canonical label at an approximate position on the
unit circle and computes the probability-weighted expectation:

| label | valence | arousal |
|---|---|---|
| happiness | +0.80 | +0.45 |
| surprise | +0.15 | +0.85 |
| fear | -0.65 | +0.75 |
| anger | -0.70 | +0.65 |
| disgust | -0.70 | +0.30 |
| contempt | -0.55 | +0.15 |
| sadness | -0.75 | -0.45 |
| neutral | 0 | 0 |

The coordinates are typical placements after [4] and [5]; they are not
measurements. A mostly-neutral distribution therefore sits near the origin,
and a confident smile moves the point into the pleasant, mildly activated
quadrant.

The raw point is smoothed for display with a time-aware exponential filter
(time constant 1.5 s). Dynamics are computed on the *raw* series resampled
to one-second steps, so the display smoothing does not inflate them:

* **Emotional inertia** is the lag-1 autocorrelation of valence [12]. In
  Kuppens and colleagues' work, higher inertia of self-reported affect was
  associated with low self-esteem and depression. AffectLab's inertia is a
  proxy computed from expression estimates, not from self-report, and should
  be read as "how sticky was the expressed affect in this window", nothing
  more.
* **Variability** is the standard deviation of valence and arousal over the
  window.
* **Switch rate** counts changes of the dominant label per minute.

## 6. Heart rate without contact: remote photoplethysmography

Each heartbeat pushes blood into the capillaries of the skin. Haemoglobin
absorbs light, so the skin's colour, particularly its green component,
oscillates very slightly with the pulse. Verkruysse and colleagues showed
in 2008 that an ordinary consumer camera under ambient light can record
this signal [13], and Poh, McDuff and Picard demonstrated a webcam
heart-rate monitor two years later [14].

AffectLab's pipeline:

1. **Region of interest.** Mean RGB of the forehead and both cheeks, defined
   by landmarks. The forehead patch stops well below the hairline so a
   fringe does not contaminate the signal.
2. **Rolling window.** Twenty seconds by default. Large head motion clears
   the window, because motion changes illumination far more than blood
   volume does.
3. **Pulse extraction**, one of three classic methods:
   * `green`: the normalised green channel [13];
   * `chrom`: chrominance signals X = 3R - 2G and Y = 1.5R + G - 1.5B,
     combined as X - alpha Y with alpha the ratio of their standard
     deviations, which cancels specular reflections and intensity changes
     [15];
   * `pos` (default): projection of the temporally normalised colour onto
     the plane orthogonal to the skin tone, computed in overlapping 1.6 s
     windows and overlap-added [16]. Wang and colleagues derived it from a
     physical model of skin reflection and showed it to be the most robust
     of the classic methods.
4. **Filtering.** Linear detrending, then a zero-phase Butterworth band-pass
   between 0.7 and 3 Hz (42 to 180 beats per minute).
5. **Spectral peak.** A Hann-windowed, zero-padded periodogram; the peak in
   band is the heart rate.
6. **Quality.** The signal-to-noise ratio of de Haan and Jeanne [15]: power
   near the peak and its first harmonic versus power elsewhere in band, in
   decibels. Above 6 dB is reported as good, above 2 dB as fair, otherwise
   poor.

Heart-rate variability (RMSSD, the root mean square of successive
inter-beat differences) is computed from peaks in the filtered waveform.
At 30 frames per second the timing of each peak is uncertain by tens of
milliseconds, so this value is marked experimental and should not be
compared with an electrocardiogram.

Known limitations: rPPG needs steady light and a still face; it degrades
with compression artefacts and low frame rates; and a meta-analysis found
systematically worse performance for darker skin tones across published
methods, a bias that comes from the physics of light absorption as well as
from unrepresentative datasets [17]. Check any reading against a pulse
oximeter before trusting it.

## 7. Blinks, alertness and breathing

The **eye aspect ratio** (EAR) is the ratio of the eye's vertical to
horizontal landmark distances [18]. It is roughly constant while the eye is
open and falls towards zero during a blink, independent of face size. A
blink is a closure of at least two frames and at most half a second, with a
threshold adapted to the person's own open-eye baseline.

**Blink rate** is a rough index of visual attention and cognitive load. In a
study of 150 healthy people the average was about 17 blinks per minute at
rest, 26 during conversation and 4.5 while reading [19]. AffectLab reports
the rate over a 60 s window and draws the resting average as a reference
line in reports.

**PERCLOS**, the proportion of time the eyes are at least 80 percent closed,
was introduced in driving-simulator research [20] and found to be the most
valid of the drowsiness measures evaluated in a US Federal Highway
Administration study [21]. AffectLab computes it from MediaPipe's eyelid
closure coefficients when available, else from EAR relative to baseline.

**Breathing rate** (experimental) is estimated from the slow vertical
oscillation of the head, band-passed between 0.1 and 0.5 Hz, over a 30 s
window. Head motion is a recognised source of respiratory signal in
camera-based monitoring [22], but the spectral resolution of a short window
is coarse.

## 8. Head pose

Yaw, pitch and roll are read from MediaPipe's facial transformation matrix.
When only landmarks are available, the pose is recovered by
Perspective-n-Point on six landmarks against a generic face model and
converted into the same frame, so both paths report zero for a frontal face
and positive yaw for a face turned towards the image's right.

## 9. What has been validated, and what has not

Tested automatically:

* the rPPG methods recover the heart rate of synthetic colour traces with
  drift, noise and irregular timestamps to within a few beats per minute;
* the blink detector counts synthetic blinks and rejects long closures;
* the FACS prototypes produce the expected label for canonical AU patterns;
* head pose recovers synthetic rotations to within 1.5 degrees;
* the whole pipeline reads a smile and a synthetic pulse from a real
  photograph (integration tests).

Not validated: agreement with certified FACS coders, with self-reported
affect, with an electrocardiogram, with a polysomnograph or with any
clinical instrument. AffectLab is a laboratory for exploring these signals,
not a validated measurement device.

## References

1. Barrett, L. F., Adolphs, R., Marsella, S., Martinez, A. M., & Pollak, S. D. (2019). Emotional expressions reconsidered: Challenges to inferring emotion from human facial movements. *Psychological Science in the Public Interest, 20*(1), 1-68. https://doi.org/10.1177/1529100619832930
2. Ekman, P., & Friesen, W. V. (1978). *Facial Action Coding System: A technique for the measurement of facial movement*. Consulting Psychologists Press.
3. Ekman, P., Friesen, W. V., & Hager, J. C. (2002). *Facial Action Coding System: The manual and investigator's guide*. Research Nexus.
4. Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology, 39*(6), 1161-1178. https://doi.org/10.1037/h0077714
5. Posner, J., Russell, J. A., & Peterson, B. S. (2005). The circumplex model of affect: An integrative approach to affective neuroscience, cognitive development, and psychopathology. *Development and Psychopathology, 17*(3), 715-734. https://doi.org/10.1017/S0954579405050340
6. Picard, R. W. (1997). *Affective computing*. MIT Press.
7. Kartynnik, Y., Ablavatski, A., Grishchenko, I., & Grundmann, M. (2019). Real-time facial surface geometry from monocular video on mobile GPUs. *arXiv:1907.06724*.
8. Grishchenko, I., Ablavatski, A., Kartynnik, Y., Raveendran, K., & Grundmann, M. (2020). Attention Mesh: High-fidelity face mesh prediction in real-time. *arXiv:2006.10962*.
9. Friesen, W. V., & Ekman, P. (1983). *EMFACS-7: Emotional Facial Action Coding System*. Unpublished manual, University of California, San Francisco.
10. Barsoum, E., Zhang, C., Canton Ferrer, C., & Zhang, Z. (2016). Training deep networks for facial expression recognition with crowd-sourced label distribution. *Proceedings of the 18th ACM International Conference on Multimodal Interaction*, 279-283. https://doi.org/10.1145/2993148.2993165
11. Serengil, S. I., & Ozpinar, A. (2021). HyperExtended LightFace: A facial attribute analysis framework. *International Conference on Engineering and Emerging Technologies (ICEET)*, 1-4. https://doi.org/10.1109/ICEET53442.2021.9659697
12. Kuppens, P., Allen, N. B., & Sheeber, L. B. (2010). Emotional inertia and psychological maladjustment. *Psychological Science, 21*(7), 984-991. https://doi.org/10.1177/0956797610372634
13. Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. *Optics Express, 16*(26), 21434-21445. https://doi.org/10.1364/OE.16.021434
14. Poh, M.-Z., McDuff, D. J., & Picard, R. W. (2010). Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. *Optics Express, 18*(10), 10762-10774. https://doi.org/10.1364/OE.18.010762
15. de Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. *IEEE Transactions on Biomedical Engineering, 60*(10), 2878-2886. https://doi.org/10.1109/TBME.2013.2266196
16. Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). Algorithmic principles of remote PPG. *IEEE Transactions on Biomedical Engineering, 64*(7), 1479-1491. https://doi.org/10.1109/TBME.2016.2609282
17. Nowara, E. M., McDuff, D., & Veeraraghavan, A. (2020). A meta-analysis of the impact of skin tone and gender on non-contact photoplethysmography measurements. *IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*, 284-285.
18. Soukupová, T., & Čech, J. (2016). Real-time eye blink detection using facial landmarks. *21st Computer Vision Winter Workshop*, Rimske Toplice, Slovenia.
19. Bentivoglio, A. R., Bressman, S. B., Cassetta, E., Carretta, D., Tonali, P., & Albanese, A. (1997). Analysis of blink rate patterns in normal subjects. *Movement Disorders, 12*(6), 1028-1034. https://doi.org/10.1002/mds.870120629
20. Wierwille, W. W., Ellsworth, L. A., Wreggit, S. S., Fairbanks, R. J., & Kirn, C. L. (1994). *Research on vehicle-based driver status/performance monitoring: Development, validation, and refinement of algorithms for detection of driver drowsiness* (Report DOT HS 808 247). National Highway Traffic Safety Administration.
21. Dinges, D. F., & Grace, R. (1998). *PERCLOS: A valid psychophysiological measure of alertness as assessed by psychomotor vigilance* (Report FHWA-MCRT-98-006). Federal Highway Administration.
22. Tarassenko, L., Villarroel, M., Guazzi, A., Jorge, J., Clifton, D. A., & Pugh, C. (2014). Non-contact video-based vital sign monitoring using ambient light and auto-regressive models. *Physiological Measurement, 35*(5), 807-831. https://doi.org/10.1088/0967-3334/35/5/807
