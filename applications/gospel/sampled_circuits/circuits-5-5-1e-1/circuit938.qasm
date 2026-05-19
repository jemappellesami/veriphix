OPENQASM 2.0;
include "qelib1.inc";
qreg q939[5];
cx q939[1],q939[0];
cx q939[3],q939[2];
rx(5*pi/4) q939[1];
cx q939[2],q939[1];
cx q939[1],q939[0];
