OPENQASM 2.0;
include "qelib1.inc";
qreg q115[3];
rx(pi/4) q115[0];
rx(3*pi/4) q115[1];
cx q115[1],q115[0];
cx q115[1],q115[2];
rx(pi/4) q115[0];
