OPENQASM 2.0;
include "qelib1.inc";
qreg q764[3];
rx(pi/2) q764[2];
cx q764[2],q764[1];
cx q764[0],q764[1];
