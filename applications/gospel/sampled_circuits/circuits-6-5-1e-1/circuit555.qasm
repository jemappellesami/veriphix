OPENQASM 2.0;
include "qelib1.inc";
qreg q556[6];
cx q556[4],q556[5];
cx q556[4],q556[3];
cx q556[3],q556[2];
cx q556[2],q556[1];
cx q556[0],q556[1];
