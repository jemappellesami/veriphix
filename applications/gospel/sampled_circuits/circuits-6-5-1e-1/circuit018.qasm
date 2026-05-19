OPENQASM 2.0;
include "qelib1.inc";
qreg q19[6];
cx q19[5],q19[4];
cx q19[3],q19[4];
cx q19[2],q19[3];
cx q19[2],q19[1];
cx q19[1],q19[0];
