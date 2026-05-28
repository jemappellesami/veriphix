OPENQASM 2.0;
include "qelib1.inc";
qreg q100[3];
rx(3*pi/2) q100[2];
cx q100[2],q100[1];
cx q100[0],q100[1];
rx(pi/4) q100[1];
