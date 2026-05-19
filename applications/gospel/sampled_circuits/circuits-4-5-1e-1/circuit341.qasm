OPENQASM 2.0;
include "qelib1.inc";
qreg q342[4];
rx(7*pi/4) q342[2];
rx(7*pi/4) q342[3];
cx q342[3],q342[2];
cx q342[2],q342[1];
cx q342[1],q342[0];
