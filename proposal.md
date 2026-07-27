**Research Proposal: Investigating Robust Overfitting in Adversarial Training**

**Principal Investigator Information**

* Name: Nicholas Q. Tran  
* Department: Department of Mathematics and Computer Science  
* Email: [ntran@scu.edu](mailto:ntran@scu.edu)  
* Phone: (408) 554-4465

**Student Information**

* Name: Kaiwen Du  
* Major: Computer Science (College of Arts and Sciences)  
* Cumulative GPA: 3.536  
* Expected Graduation Year: 2028  
* Relevant Experience and Interests: Python, Software Development, Git/Version Control, Linear Algebra, Calculus; interest in AI safety and machine learning vulnerabilities.

**The Problem:** Deep neural networks are vulnerable to adversarial attacks, small imperceptible input perturbations that cause confident misclassification. Goodfellow et al. (2014) traced this vulnerability to the linear nature of modern networks and introduced adversarial training as a defense. However, Rice et al. (2020) identified a critical failure mode in PGD-based adversarial training, one of the strongest known empirical defenses: robust overfitting, where robust test error rises during later training even as training loss falls. Standard regularization methods fail to resolve it, and its root cause remains unknown.

**Our Hypothesis:** Adversarial training causes models to memorize specific perturbation patterns rather than learning a generalizable robust decision boundary, and this memorization is the root cause of robust overfitting. This project makes two contributions: an independent replication of Rice et al.'s findings, which is valuable given reproducibility concerns in empirical ML, and a fine-grained empirical characterization of the overfitting point itself, which Rice et al. documented but did not isolate as a standalone object of study. Pinpointing when memorization overtakes genuine robustness learning provides a diagnostic foundation for future work on safeguards against robust overfitting.

**Research Goals, Methods, and Technical Approach:** This project focuses on building a rigorous empirical foundation for studying robust overfitting across two goals. First, we will reproduce the adversarial training setup from Rice et al. (2020) using PreActResNet-18 on CIFAR-10 with PGD-10, confirming that robust overfitting occurs as reported. Second, we will track training and test robust accuracy across epochs to precisely locate when the gap between training and test robustness emerges and grows.

**Tangible Deliverables:**

* A written literature review summarizing prior work on adversarial training and robust overfitting.  
* A public GitHub repository containing fully reproducible training and evaluation code.  
* Learning curves tracking training and test robust accuracy to precisely identify the overfitting point.  
* A final report documenting our methodology and findings, prepared for publication.

**8 Week Summer Schedule:**

| Week \# | Objectives |
| :---- | :---- |
| Week 1 | Review, summarize prior literature on Goodfellow et al. (2014) and Rice et al. (2020). |
| Week 2 | Set up software environment, cloud computing resources, and model implementations. |
| Week 3 | Implement PGD-based adversarial training pipeline and verify on a small test run. |
| Week 4 | Run full baseline adversarial training on CIFAR-10, verify overfitting is reproducible. |
| Week 5 | Track training and test robust accuracy across epochs to identify the overfitting point. |
| Week 6 | Run additional training runs with different random seeds to verify overfitting point consistency. |
| Week 7 | Analyze results, produce learning curve visualizations, and draft findings. |
| Week 8 | Complete the final report and prepare for publication and poster presentation. |

**Responsible AI Scope:** This project addresses the Safety and Accountability dimensions of Responsible AI. Robust overfitting creates a concrete accountability gap: a model can pass every adversarial robustness benchmark in a lab setting yet fail against slightly different real-world attacks. In high-stakes deployments such as medical image analysis or content moderation, this gap between certified performance and actual reliability poses direct harm to the people those systems serve. Without knowing the exact point at which a model transitions from learning robustness to memorizing perturbation patterns, it is impossible to design reliable safeguards or set meaningful accountability standards for deployed AI systems.

**Impact on Faculty Research:** This project builds directly on Dr. Tran's prior work on Digital Image Watermarking, which studied how small, structured image modifications interact with machine learning systems. Adversarial perturbations are mathematically similar objects: both are imperceptible to human observers yet meaningful to models. A robustly overfitting model that memorizes specific perturbation patterns rather than learning genuine robustness may similarly fail against novel watermarking schemes. By identifying precisely when this memorization occurs, this project produces technical grounding that directly extends Dr. Tran's research agenda into adversarial machine learning.

**Availability and Collaborations:** Kaiwen and Dr. Tran are both available full-time for the 8-week summer schedule. Dr. Tran will supervise through weekly one-on-one meetings to review experimental results, provide feedback on written work, and guide research decisions. While no formal interdisciplinary collaborations are involved, the findings have direct relevance beyond computer science: results will inform how model robustness certifications should be interpreted in legal, policy, and organizational contexts where AI accountability is at stake.

**Dissemination of Results:**

* Internal: Poster presentation at the Santa Clara University Research Symposium.  
* External: AAAI Undergraduate Consortium or a similar student track at a major AI conference.  
* Code Sharing: Publish all code publicly on GitHub for reproducibility.

**References:**

* Goodfellow, I. J., Shlens, J., and Szegedy, C. (2014). Explaining and Harnessing Adversarial Examples. *ICLR.*  
* Rice, L., Wong, E., and Kolter, J. Z. (2020). Overfitting in adversarially robust deep learning. *ICML.*

**Proposed Summer 2026 Budget**

**Student Wages and Benefits:** Funding for one undergraduate student (Kaiwen Du) working full-time for 8 weeks during the Summer 2026 quarter. This will be calculated using the standardized hourly wage rate provided by the Responsible AI program.

**Compute Budget:** $200

Cloud GPU credits for approximately 200 hours of adversarial training experiments, hosted on Lambda Labs at \~$0.75/hr (A10G instance). This estimate is grounded in reported training times from Rice et al. (2020), which used a single GeForce RTX 2080 Ti for PGD adversarial training on CIFAR-10. A comparable cloud instance requires approximately 5 hours per ResNet-18 training run and up to 12 hours for wider architectures, across multiple runs with different random seeds to ensure reproducibility. The full breakdown is as follows:

| Item | Estimate | Cost |
| :---- | :---- | :---- |
| Baseline \+ validation runs (ResNet-18) | 8 runs × 5 hrs | $30 |
| Overfitting point tracking experiments | 5 runs × 5 hrs | $19 |
| Seed variation runs | 6 runs × 5 hrs | $23 |
| Debug and re-run buffer | \~83 hrs | $62 |
| Wide ResNet experiments (if time allows) | 5 runs × 12 hrs | $45 |
| **Total** | \~204 hrs @ $0.75/hr | \~$153 |

**Total Funding Requested:** Standard 8-week student wages plus $200 compute budget.

**Compliance Note:** No funds are requested for equipment, supplies, travel, faculty salary, or tuition. All funds will be used during the summer period before October 1, 2026\.