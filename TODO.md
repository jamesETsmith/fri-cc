## Update Amps
- [X] 0101 O^4V^2 `t2new += lib.einsum('klij,klab->ijab', Woooo, tau)`
- [X] 2323 O^2V^4 `t2new += lib.einsum('abcd,ijcd->ijab', Wvvvv, tau)`
- [X] 1302 O^3V^3 `tmp  = 2*lib.einsum('akic,kjcb->ijab', Wvoov, t2)`
- [X] 1202 O^3V^3 `tmp -=   lib.einsum('akci,kjcb->ijab', Wvovo, t2)`
- [X] 1303 O^3V^3 `tmp = lib.einsum('akic,kjbc->ijab', Wvoov, t2)`
- [X] 1203 O^3V^3 `tmp = lib.einsum('bkci,kjac->ijab', Wvovo, t2)`

## Intermediates

### Woooo
- [ ] 1323 O^4V^2 `Wklij += lib.einsum('kcld,ijcd->klij', eris_ovov, t2)`
- [ ] `Wklij += lib.einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)`

### Wvoov
- [ ] 0112 O^3V^3 `Wakic -= 0.5*lib.einsum('ldkc,ilda->akic', eris_ovov, t2)`
- [ ] 0313 O^3V^3 `Wakic -= 0.5*lib.einsum('lckd,ilad->akic', eris_ovov, t2)`
- [ ] `Wakic -= lib.einsum('ldkc,id,la->akic', eris_ovov, t1, t1)`

### Wovov
- [ ] 0312 O^3V^3 `Wakci -= 0.5*lib.einsum('lckd,ilda->akci', eris_ovov, t2)`
- [ ] `Wakci -= lib.einsum('lckd,id,la->akci', eris_ovov, t1, t1)`