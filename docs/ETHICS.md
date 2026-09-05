# Ethics, limits and responsible use

AffectLab measures faces and bodies. That is a sensitive thing to do, and
the science, the law and common decency all put limits on what the numbers
mean and where they may be used. This page is short on purpose. Please read
it before pointing the camera at anyone but yourself.

## What AffectLab is not

* **Not a lie detector.** Nothing here measures truthfulness.
* **Not a mind reader.** A label such as "anger" means "the face moved in a
  way an algorithm associates with anger prototypes". People frown when they
  concentrate, smile when they are embarrassed, and keep still when they are
  furious.
* **Not a medical device.** Heart rate, heart-rate variability, breathing
  rate, blink rate and PERCLOS are estimated from an ordinary camera and
  have not been validated against clinical instruments. They must not be
  used to diagnose, treat or monitor any condition, or to decide whether
  someone is fit to drive, work or study.
* **Not a screening or surveillance tool.** It has no place in hiring,
  performance reviews, exam proctoring, classroom monitoring, border control
  or policing.

## What the science says

The largest review of the evidence [1] found that facial movements are
neither reliable (the same emotion is expressed in many ways), nor specific
(the same movement appears in many states), nor generalisable across
cultures and contexts. Facial expression is *a* signal about affect, weak on
its own and heavily dependent on context.

AffectLab tries to be honest about this by:

* reporting Action Units (descriptions of movement) separately from emotion
  labels (interpretations);
* exposing the evidence behind each rule-based label;
* preferring continuous valence and arousal to hard categories;
* attaching a quality grade to every physiological estimate;
* saying "indicative" and "experimental" where it applies, on screen and in
  reports.

Automatic expression recognisers also inherit the biases of their training
data. FER-style datasets over-represent some ages, ethnicities and posed
expressions. Remote photoplethysmography performs systematically worse on
darker skin tones across published methods [2]. Test AffectLab on yourself
and on people who look different from you before believing any of it about
anyone.

## What the law says

Laws differ by jurisdiction and change quickly. Two examples you should know
about:

* **European Union, AI Act.** Since 2 February 2025, Article 5(1)(f)
  prohibits placing on the market or using AI systems that infer the
  emotions of a natural person in the areas of the **workplace** and
  **education institutions**, except for medical or safety reasons. Emotion
  recognition systems permitted elsewhere carry transparency duties (people
  must be told), and many uses are classed as high risk. Fines reach 35
  million euros or 7 percent of worldwide turnover.
* **Data protection.** Facial geometry and physiological signals derived
  from a person's body are biometric or health-related data. Under the GDPR
  and similar laws they are special-category data that generally require
  explicit consent and a clear purpose, and several US states regulate
  biometric identifiers separately.

If you are unsure whether a use is lawful where you are, assume it is not
until a qualified person tells you otherwise.

## Privacy by design

* All processing happens on your machine. AffectLab makes network requests
  only to download the model files listed in `affectlab models list`, and
  verifies each against a published checksum. No frames, features or results
  are ever uploaded.
* Recorded sessions (`--record`) contain numbers, not images. They are still
  personal data about the person in front of the camera. Store them
  accordingly and delete them when you are done.
* The `b` key (or `--blur`) pixelates the face in the display and in any
  annotated video you save.
* Nothing is written to disk unless you ask for it.

## Before you use it on someone else

1. Tell them what is being measured, why, and what happens to the data.
2. Get a clear yes. Silence is not consent; neither is an employment
   contract or a class enrolment.
3. Let them see the screen, stop at any time, and have the data deleted.
4. Do not use the output to make or justify any decision about them.
5. Never claim the numbers reveal what someone "really" feels.

AffectLab is released under the MIT licence, which does not restrict fields
of use. These are requests, not licence terms. Please honour them anyway.

## References

1. Barrett, L. F., Adolphs, R., Marsella, S., Martinez, A. M., & Pollak, S. D. (2019). Emotional expressions reconsidered: Challenges to inferring emotion from human facial movements. *Psychological Science in the Public Interest, 20*(1), 1-68.
2. Nowara, E. M., McDuff, D., & Veeraraghavan, A. (2020). A meta-analysis of the impact of skin tone and gender on non-contact photoplethysmography measurements. *IEEE/CVF CVPR Workshops*.
3. Regulation (EU) 2024/1689 (Artificial Intelligence Act), Article 5(1)(f) and Article 50.
