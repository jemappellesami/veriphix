OPENQASM 2.0;
include "qelib1.inc";
qreg q225[3];
cx q225[2],q225[1];
rz(3*pi/2) q225[2];
cx q225[2],q225[1];
cx q225[1],q225[0];
