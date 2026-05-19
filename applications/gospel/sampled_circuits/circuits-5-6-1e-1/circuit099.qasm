OPENQASM 2.0;
include "qelib1.inc";
qreg q100[5];
cx q100[3],q100[4];
cx q100[3],q100[2];
cx q100[2],q100[1];
cx q100[1],q100[0];
rx(pi/4) q100[1];
