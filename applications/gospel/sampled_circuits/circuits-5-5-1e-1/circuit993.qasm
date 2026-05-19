OPENQASM 2.0;
include "qelib1.inc";
qreg q994[5];
rx(3*pi/2) q994[0];
cx q994[3],q994[4];
cx q994[2],q994[3];
cx q994[2],q994[1];
cx q994[0],q994[1];
