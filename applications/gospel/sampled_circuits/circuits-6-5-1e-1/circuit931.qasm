OPENQASM 2.0;
include "qelib1.inc";
qreg q932[6];
cx q932[5],q932[4];
cx q932[4],q932[3];
cx q932[2],q932[3];
cx q932[2],q932[1];
cx q932[1],q932[0];
