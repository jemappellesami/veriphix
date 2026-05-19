OPENQASM 2.0;
include "qelib1.inc";
qreg q369[5];
rx(pi/4) q369[4];
cx q369[4],q369[3];
cx q369[2],q369[3];
cx q369[2],q369[1];
cx q369[1],q369[0];
