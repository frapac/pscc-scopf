
# Editor
- [ ] Compare with MadNCL-CPU+ExaModels, MadNCL-CPU+JuMP, MadNCL-CPU+JuMP+SymbolicAD

# Reviewer 1
- [ ] Explain why product w_1w_2 <= 0 is equivalent to the MPCC (hard to interpret currently)
- [ ] Improve technical description for Section IV
    - [ ] Definition of the index sets
    - [ ] Sign of r and t
    - [ ] Update rule for rho
    - [ ] Definition of variable x
- [ ] Explain better the signed Cholesky factorization and MadNLP (how does it differ from other solver?)
- [ ] Include information about constraint violation for equality and inequality

# Reviewer 2
- [ ] Motivate why MINLP reformulation is intractable
- [ ] Condition number of the Newton system as function of \rho for ACOPF
- [ ] Comments better why Knitro needs more iterations, and why MadNLP-CPU needs more iterations than MadNCL-GPU

# Reviewer 3
- [ ] Solve instance from the GO-competition (allow for infeasible contingencies)
- [ ] Remove III.B and Lagrangian in III.A? Also easier description of strong stationarity
- [ ] Clarify contingency screening procedure (low or high infeasibility)
- [ ] Provide more details about the base-case AC OPF model
- [ ] Motivate better why problem violates MFCQ
- [ ] Issue with Eq (11) (w1, w2) \geq 0
- [ ] Clarify inner and outer iterations in IV.A
- [ ] Clarify that solver can converge to a spurious locally infeasible solution
