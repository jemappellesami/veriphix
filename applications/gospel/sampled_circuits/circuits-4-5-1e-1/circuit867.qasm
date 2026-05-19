OPENQASM 2.0;
include "qelib1.inc";
qreg q868[4];
rx(3*pi/4) q868[3];
cx q868[3],q868[2];
cx q868[1],q868[2];
cx q868[1],q868[0];
