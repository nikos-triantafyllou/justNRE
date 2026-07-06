   <div align="center">
     <img src="images/justNRE.png">
   </div>

An Occam's razor inspired Neural Ratio Estimation code for simulation-based inference
======================================================================================

This is a code to perform the Neural Ratio Estimation (NRE) flavor of Simulation Based Inference (SBI).

When given data $d$ and parameters $\theta$, the goal is to find the posterior:

$$ p(\boldsymbol\theta|\boldsymbol d) = \frac{p(\boldsymbol d|\boldsymbol \theta)\cdot p(\boldsymbol \theta)}{p(\boldsymbol d)} $$


\begin{equation}
    r=\frac{p(\boldsymbol{x} \mid \boldsymbol{\theta})}{p(\boldsymbol{x})} 
    = \frac{p(\boldsymbol{\theta} \mid \boldsymbol{x})}{p(\boldsymbol{\theta})} 
    = \frac{p(\boldsymbol{x}, \boldsymbol{\theta})}{p(\boldsymbol{x}) p(\boldsymbol{\theta})}.
\end{equation}


Classifier:
\begin{equation}
    \tilde{p}(\boldsymbol{x}, \boldsymbol{\theta} \mid y) =
    \begin{cases} 
        p(\boldsymbol{x}, \boldsymbol{\theta}) & \text{if } y = 1, \\
        p(\boldsymbol{x}) p(\boldsymbol{\theta}) & \text{if } y = 0.
    \end{cases}
\end{equation}


How to get the ratio from the classifier:
$$\frac{p(\boldsymbol{x}, \boldsymbol{\theta})}{p(\boldsymbol{x}) p(\boldsymbol{\theta})}
= \frac{\tilde{p}(\boldsymbol{x}, \boldsymbol{\theta} \mid y=1)}{\tilde{p}(\boldsymbol{x}, \boldsymbol{\theta} \mid y=0)}
= \frac{\tilde{p}(\boldsymbol{x}, \boldsymbol{\theta}, y=1)}{\tilde{p}(\boldsymbol{x}, \boldsymbol{\theta}, y=0)} \\
= \frac{\tilde{p}(y=1 \mid \boldsymbol{x}, \boldsymbol{\theta})}{\tilde{p}(y=0 \mid \boldsymbol{x}, \boldsymbol{\theta})}
= \frac{\tilde{p}(y=1 \mid \boldsymbol{x}, \boldsymbol{\theta})}{1 - \tilde{p}(y=1 \mid \boldsymbol{x}, \boldsymbol{\theta})}.
$$

