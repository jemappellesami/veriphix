OPENQASM 2.0;
include "qelib1.inc";
qreg q518[7];
cx q518[5],q518[4];
cx q518[3],q518[4];
cx q518[3],q518[2];
cx q518[2],q518[1];
cx q518[1],q518[0];
