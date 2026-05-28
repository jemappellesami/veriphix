OPENQASM 2.0;
include "qelib1.inc";
qreg q720[3];
rx(3*pi/4) q720[0];
cx q720[0],q720[1];
cx q720[1],q720[2];
rx(pi/4) q720[0];
