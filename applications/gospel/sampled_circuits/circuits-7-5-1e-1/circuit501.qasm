OPENQASM 2.0;
include "qelib1.inc";
qreg q502[7];
cx q502[4],q502[5];
cx q502[4],q502[3];
cx q502[2],q502[3];
cx q502[2],q502[1];
cx q502[0],q502[1];
