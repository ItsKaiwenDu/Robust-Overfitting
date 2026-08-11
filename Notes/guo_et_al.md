# Notes: "Low Frequency Adversarial Perturbation" (Guo, Frank & Weinberger, 2019)

## 1. Summary

This paper focuses on making black-box adversarial attacks, meaning attacks where the attacker cannot see the model's gradients and can only query it for outputs, much more query-efficient. The authors propose restricting the search for adversarial perturbations to a low-frequency subspace using the discrete cosine transform (DCT), the same transform used in JPEG compression. Instead of letting an attack algorithm search or sample across the full, high-dimensional pixel space, they constrain it to a much smaller subspace made only of low-frequency patterns. They show this simple restriction is compatible with existing black-box attack frameworks and consistently reduces the number of model queries needed by 2 to 4 times. It also lets attacks bypass certain defenses that filter out high-frequency noise, and the authors demonstrate a real-world version of this by tricking the Google Cloud Vision platform using an unusually small number of queries.

## 2. Key Terms

- **Black-Box Attack**: An attack setting where the adversary can only send inputs to a model and observe its outputs (such as a label or confidence score), with no access to its gradients, architecture, or training data. This contrasts with white-box attacks, where the attacker has full internal access to the model.
- **Query Complexity**: The number of times an attack algorithm needs to call the target model in order to successfully produce an adversarial example. Lower query complexity means a cheaper and faster attack.
- **Discrete Cosine Transform (DCT)**: A mathematical transform that rewrites an image as a combination of cosine wave patterns at different frequencies, separating an image's low-frequency structure from its high-frequency detail. It is invertible, meaning you can convert back and forth between pixel space and frequency space without losing information.
- **Boundary Attack**: A black-box attack that starts from a clearly adversarial image and repeatedly moves it closer to the original image while checking that it remains adversarial at each step, gradually shrinking the size of the perturbation.
- **NES (Natural Evolution Strategies) Attack**: A black-box attack that estimates a useful search direction by sampling many small random perturbations around the current image, checking how each one affects the model's loss, and combining those results into an approximate gradient direction.

## 3. Black Box Attacks Are Expensive

In the white-box setting, an attacker has full access to a model, including its gradients, so it can directly calculate which small change to an image would trick the model. This tends to require very few queries, sometimes as few as 10 gradient evaluations.

In the black-box setting, and this is where attackers usually at, has no access to the model's internal workings. Instead, black-box attacks generally work by randomly sampling directions in the image's pixel space and checking, through repeated queries, whether moving in that direction makes progress toward tricking the model. Because natural images have very high dimensionality (a single ImageNet image can have well over 100,000 pixel values across its color channels), most randomly sampled directions in this space are not useful, so a large number of queries end up wasted. As a result, black-box attacks on models like ResNet on ImageNet can require on the order of 10,000 to 100,000 queries to succeed, which becomes both slow and, in real-world settings, potentially expensive or easily detectable.

## 4. Idea: Searching in Low-Frequency Space

The paper's central proposal is to stop searching the full pixel space and instead restrict the search entirely to a low-frequency subspace.

The motivation comes from image compression, particularly JPEG. JPEG works well because most of the meaningful, content-defining information in a natural image is concentrated in its low-frequency components, while high-frequency components tend to correspond to fine detail or noise. Based on this, the authors reason that convolutional neural networks likely respond especially strongly to these same low-frequency patterns, since that is where the class-defining structure of an image tends to live. It is important to note that the authors present this specifically as a plausible assumption motivating their approach, not as something they mathematically prove. They describe it as being "plausible to assume," and the rest of the paper is dedicated to testing this assumption empirically rather than proving it analytically beforehand.

To operationalize this, they use the DCT, which converts an image from pixel space into frequency space, where each entry represents the strength of a particular cosine wave pattern, with lower-indexed entries corresponding to lower, smoother frequencies. By only allowing perturbations to have nonzero values in the low-frequency entries of this representation, and setting a ratio parameter r to control how much of the frequency spectrum is kept, they can sample or optimize entirely within this reduced subspace. For example, restricting to a ratio of r = 1/8 reduces the dimensionality of the search space by a factor of 64, while, according to their results, still containing enough adversarial directions to reach the same success rate as searching the full pixel space.

The authors support this empirically by comparing the success rate of purely random perturbations in normal RGB pixel space versus in this low-frequency DCT (LF-DCT) subspace. They find that random noise sampled in the low-frequency subspace is dramatically more likely to produce a successful adversarial perturbation than the same amount of random noise sampled in full pixel space, which supports their claim that adversarial directions are denser, or more common, within the low-frequency region.

## 5. Applying by Modifying Two Existing Attacks

A major strength the authors highlight is that this low-frequency restriction does not require inventing an entirely new attack. Instead, it can be added into existing black-box attack algorithms with a small modification to how they sample random noise.

They demonstrate this on two well-known black-box attacks:

- **Low Frequency Boundary Attack (LF-BA)**: The original boundary attack repeatedly samples a random noise direction from a standard Gaussian distribution, checks whether adding it keeps the image adversarial, and gradually contracts the perturbation toward the original image. The low-frequency version simply replaces this Gaussian sampling step with sampling from the low-frequency DCT subspace instead. Since low-frequency directions are more likely to remain adversarial, fewer wasted queries are needed at each step.
- **Low Frequency NES (LF-NES)**: The original NES attack estimates a useful gradient-like direction by sampling a batch of small Gaussian noise vectors around the current image and weighting them by how much each one increases the model's loss. The low-frequency version replaces the Gaussian noise vectors with noise vectors sampled from the low-frequency DCT subspace, while keeping the rest of the underlying gradient estimation procedure unchanged.

In both cases, the underlying attack logic stays the same. Only the distribution used to generate candidate perturbations changes, which is part of why the authors describe this technique as easy to incorporate into other existing black-box attack frameworks as well.

## 6. Results: Query Efficiency & Breaking Defenses

The paper reports several concrete results evaluated on ImageNet using a pretrained ResNet-50 model:

- **Query reduction for LF-BA**: The low-frequency boundary attack required a median of 1,128 queries to succeed, compared to 4,020 median queries for the original boundary attack, an almost 4 times reduction.
- **Query reduction for LF-NES**: The low-frequency NES attack required a median of about 12,444 queries, compared to 22,389 median queries for the original NES attack, roughly a 2 times reduction.
- **Circumventing image transformation defenses**: Some defenses work by applying a denoising transformation, such as JPEG compression or reducing an image's bit depth, before feeding an input into the model, which is intended to remove adversarial noise before it reaches the classifier. The original boundary attack made no progress at all against these defenses within the query budget tested. The low-frequency boundary attack was able to consistently defeat both defenses and still reach a low perturbation error, since these defenses mainly target high-frequency noise and do not interfere much with low-frequency signal.
- **Attacking Google Cloud Vision**: As a real-world demonstration, the authors attacked Google Cloud Vision, a publicly available commercial image classification service, using the low-frequency boundary attack. It successfully changed the platform's top predicted concept for an image using approximately 1,000 model queries, which the authors describe as an unprecedented reduction, and something the original, non-low-frequency version of the attack was not able to do within a comparable number of queries.

Overall, the results support the paper's central claim: restricting the search space to low-frequency perturbations does not meaningfully limit an attack's power, but it substantially reduces the number of queries required and can bypass certain existing defenses.

## 7. Works Cited

Guo, Chuan, Jared S. Frank, and Kilian Q. Weinberger. "Low Frequency Adversarial Perturbation." *Proceedings of the 35th Conference on Uncertainty in Artificial Intelligence (UAI 2019)*, 2019.