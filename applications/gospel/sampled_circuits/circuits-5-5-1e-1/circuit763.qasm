OPENQASM 2.0;
include "qelib1.inc";
qreg q764[5];
cx q764[4],q764[3];
cx q764[2],q764[3];
cx q764[2],q764[1];
cx q764[1],q764[0];
