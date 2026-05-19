OPENQASM 2.0;
include "qelib1.inc";
qreg q445[5];
rx(7*pi/4) q445[4];
cx q445[3],q445[4];
cx q445[3],q445[2];
cx q445[2],q445[1];
cx q445[1],q445[0];
