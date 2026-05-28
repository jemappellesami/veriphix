OPENQASM 2.0;
include "qelib1.inc";
qreg q60[3];
rx(5*pi/4) q60[1];
rx(5*pi/4) q60[2];
cx q60[1],q60[2];
cx q60[0],q60[1];
rx(pi/4) q60[1];
