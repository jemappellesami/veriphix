OPENQASM 2.0;
include "qelib1.inc";
qreg q540[3];
rx(7*pi/4) q540[0];
rx(pi/4) q540[2];
cx q540[1],q540[0];
cx q540[2],q540[1];
rx(pi/4) q540[0];
