   <div align="center">
     <img src="images/justNRE.png">
   </div>

An Occam's razor inspired Neural Ratio Estimation code for simulation-based inference
======================================================================================

This is a code to perform the Neural Ratio Estimation (NRE) flavor of Simulation Based Inference (SBI).

When given data $d$ and parameters $\theta$,the goal is to find the posterior:

$$ p(\boldsymbol\theta|\boldsymbol d) = \frac{p(\boldsymbol d|\boldsymbol \theta)\cdot p(\boldsymbol \theta)}{p(\boldsymbol d)} $$

