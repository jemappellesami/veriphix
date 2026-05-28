OPENQASM 2.0;
include "qelib1.inc";
qreg q172[3];
rx(pi/2) q172[1];
rx(pi/2) q172[2];
rz(pi/4) q172[2];
cx q172[2],q172[1];
cx q172[0],q172[1];
