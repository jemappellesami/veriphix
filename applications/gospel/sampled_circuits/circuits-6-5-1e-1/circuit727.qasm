OPENQASM 2.0;
include "qelib1.inc";
qreg q728[6];
cx q728[1],q728[2];
cx q728[2],q728[3];
cx q728[2],q728[1];
cx q728[1],q728[0];
