OPENQASM 2.0;
include "qelib1.inc";
qreg q580[4];
rx(pi) q580[3];
rz(3*pi/2) q580[3];
cx q580[2],q580[3];
cx q580[2],q580[1];
cx q580[1],q580[0];
