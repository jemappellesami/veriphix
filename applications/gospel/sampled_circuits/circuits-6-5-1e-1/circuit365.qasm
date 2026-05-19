OPENQASM 2.0;
include "qelib1.inc";
qreg q366[6];
cx q366[5],q366[4];
cx q366[3],q366[4];
cx q366[3],q366[2];
cx q366[2],q366[1];
cx q366[0],q366[1];
