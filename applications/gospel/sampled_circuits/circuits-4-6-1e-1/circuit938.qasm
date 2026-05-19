OPENQASM 2.0;
include "qelib1.inc";
qreg q939[4];
cx q939[3],q939[2];
rx(pi/4) q939[2];
cx q939[1],q939[2];
cx q939[1],q939[0];
rx(pi/4) q939[1];
